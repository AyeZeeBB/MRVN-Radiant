/* -------------------------------------------------------------------------------

   Copyright (C) 2022-2024 MRVN-Radiant and contributors.
   For a list of contributors, see the accompanying CONTRIBUTORS file.

   This file is part of MRVN-Radiant.

   MRVN-Radiant is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   MRVN-Radiant is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GtkRadiant; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

   ------------------------------------------------------------------------------- */


#include "../remap.h"
#include "../bspfile_abstract.h"
#include "apex_legends_collision_types.h"
#include <ctime>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cstring>

/*
    BVH4 Collision System for Apex Legends BSP
    
    Based on reverse engineering of Apex Legends r5apex.exe via IDA:
    - CollBvh4Node_s: 64-byte BVH4 nodes with packed bounds and metadata
    - Child types: 0=Node, 1=None, 2=Empty, 3=Bundle, 4=TriStrip, 5=Poly3, 6=Poly4, 7=Poly5+, 8=ConvexHull, 9=StaticProp, 10=Heightfield
    - Packed vertices: 6 bytes (int16 x,y,z), decoded as: world = origin + (int16 << 16) * scale
    - Node bounds: int16[24] in SOA format, same decode formula
    
    Key insight from CollPoly_Visit_5:
    - baseVertex stored in header bytes 2-3 (not shifted)
    - Game computes: running_base = baseVertex << 10
    - Per-triangle: v0 = running_base + offset, v1 = v0 + 1 + delta1, v2 = v0 + 1 + delta2
    - running_base updated to v0 after each triangle
    
    Vertex decode from IDA:
    - packed_verts[idx] at byte offset idx * 6
    - SSE: cvtepi32_ps(unpacklo_epi16(0, int16x3)) produces (int16 << 16)
    - world.xyz = origin.xyz + (int16 << 16) * origin.w
*/

namespace {
    // BVH4 child types from IDA analysis of CollBvh_VisitLeafs (CollBvh_VisitNodes_r)
    // These match ApexLegends::Collision::ChildType enum in apex_legends_collision_types.h
    constexpr int BVH4_TYPE_NODE        = 0;  // Internal BVH node - recurse
    constexpr int BVH4_TYPE_NONE        = 1;  // Empty child slot
    constexpr int BVH4_TYPE_EMPTY       = 2;  // Empty leaf
    constexpr int BVH4_TYPE_BUNDLE      = 3;  // Bundle of leaf entries
    constexpr int BVH4_TYPE_TRISTRIP    = 4;  // Triangle strip (FLOAT vertices)
    constexpr int BVH4_TYPE_POLY3       = 5;  // Triangle polygon (PACKED int16 vertices)
    constexpr int BVH4_TYPE_POLY4       = 6;  // Quad polygon (PACKED int16 vertices)
    constexpr int BVH4_TYPE_POLY5PLUS   = 7;  // 5+ vertex polygon (PACKED int16 vertices)
    constexpr int BVH4_TYPE_CONVEXHULL  = 8;  // Convex hull (brush collision)
    constexpr int BVH4_TYPE_STATICPROP  = 9;  // Static prop reference
    constexpr int BVH4_TYPE_HEIGHTFIELD = 10; // Heightfield terrain

    // Component-wise min/max for Vector3
    inline Vector3 Vec3Min(const Vector3& a, const Vector3& b) {
        return Vector3(
            std::min(a.x(), b.x()),
            std::min(a.y(), b.y()),
            std::min(a.z(), b.z())
        );
    }

    inline Vector3 Vec3Max(const Vector3& a, const Vector3& b) {
        return Vector3(
            std::max(a.x(), b.x()),
            std::max(a.y(), b.y()),
            std::max(a.z(), b.z())
        );
    }

    // Maximum triangles per leaf before splitting
    constexpr int MAX_TRIS_PER_LEAF = 8;

    // Maximum depth for BVH tree
    constexpr int MAX_BVH_DEPTH = 32;

    // Minimum triangle edge length to prevent degenerate triangles
    constexpr float MIN_TRIANGLE_EDGE = 0.1f;

    // Minimum triangle area to prevent sliver triangles
    constexpr float MIN_TRIANGLE_AREA = 0.01f;

    // NOTE: Leaf data structures are now defined in apex_legends_collision_types.h
    // We use them directly from the ApexLegends::Collision namespace

    // Triangle for collision
    struct CollisionTri_t {
        Vector3 v0, v1, v2;
        Vector3 normal;
        int contentFlags;
        int surfaceFlags;
    };

    // Convex hull for collision (brushes)
    struct CollisionHull_t {
        std::vector<Vector3> vertices;
        std::vector<std::array<int, 3>> faces;  // Triangle indices
        std::vector<Plane3f> planes;
        int contentFlags;
        Vector3 origin;
        float scale;
    };

    // Static prop reference for collision
    struct CollisionStaticProp_t {
        uint32_t propIndex;
        MinMax bounds;
    };

    // Heightfield patch for terrain collision
    struct CollisionHeightfield_t {
        uint8_t cellX, cellY;
        std::array<int16_t, 16> heights;
        uint8_t materialIndex;
        MinMax bounds;
    };

    // Internal BVH node for building
    struct BVHBuildNode_t {
        MinMax bounds;
        int childIndices[4] = { -1, -1, -1, -1 };
        int childTypes[4] = { BVH4_TYPE_NONE, BVH4_TYPE_NONE, BVH4_TYPE_NONE, BVH4_TYPE_NONE };
        std::vector<int> triangleIndices;      // For poly/tristrip leaves
        std::vector<int> hullIndices;          // For convex hull leaves
        std::vector<int> staticPropIndices;    // For static prop leaves
        std::vector<int> heightfieldIndices;   // For heightfield leaves
        bool isLeaf = false;
        int contentFlags = CONTENTS_SOLID;
        int preferredLeafType = BVH4_TYPE_CONVEXHULL;
    };

    // Global collision data for current model
    std::vector<CollisionTri_t> g_collisionTris;
    std::vector<CollisionHull_t> g_collisionHulls;
    std::vector<CollisionStaticProp_t> g_collisionStaticProps;
    std::vector<CollisionHeightfield_t> g_collisionHeightfields;
    
    // Built BVH nodes
    std::vector<BVHBuildNode_t> g_bvhBuildNodes;
    
    // Global BVH origin and scale for current model
    Vector3 g_bvhOrigin = Vector3(0, 0, 0);
    float g_bvhScale = 1.0f / 65536.0f;  // Default scale
    
    // Base vertex index for current model's packed vertices
    uint32_t g_modelPackedVertexBase = 0;
    
    /*
        EmitPackedVertex
        Encodes a world position as a packed int16x3 vertex and adds to the lump
        Returns the index into the packed vertices array
        
        From IDA (CollPoly_Visit_5):
        - SSE: cvtepi32_ps(unpacklo_epi16(0, int16x3)) produces (int16 << 16)
        - world.xyz = origin.xyz + (int16 << 16) * origin.w
        - So: world = origin + int16 * 65536 * scale
        
        To encode: int16 = (world - origin) / (scale * 65536)
    */
    uint32_t EmitPackedVertex(const Vector3& worldPos) {
        // The game decodes as: worldPos = origin + (int16 << 16) * scale
        // Which equals: worldPos = origin + int16 * 65536 * scale
        // So we encode as: int16 = (worldPos - origin) / (scale * 65536)

        float invScaleFactor = 1.0f / (g_bvhScale * 65536.0f);

        float px = (worldPos.x() - g_bvhOrigin.x()) * invScaleFactor;
        float py = (worldPos.y() - g_bvhOrigin.y()) * invScaleFactor;
        float pz = (worldPos.z() - g_bvhOrigin.z()) * invScaleFactor;

        ApexLegends::PackedVertex_t vert;
        vert.x = static_cast<int16_t>(std::clamp(px, -32768.0f, 32767.0f));
        vert.y = static_cast<int16_t>(std::clamp(py, -32768.0f, 32767.0f));
        vert.z = static_cast<int16_t>(std::clamp(pz, -32768.0f, 32767.0f));

        uint32_t idx = static_cast<uint32_t>(ApexLegends::Bsp::packedVertices.size());
        ApexLegends::Bsp::packedVertices.push_back(vert);

        return idx;
    }

    /*
        PackBoundsToInt16
        Converts float bounds to int16_t format used by Apex BSP
        The format packs child bounds into [axis][min/max][child] layout

        Game decodes as: worldPos = origin + (int16 * 65536) * scale
        So we encode as: int16 = (worldPos - origin) / (scale * 65536)
    */
    void PackBoundsToInt16(const MinMax bounds[4], int16_t outBounds[24]) {
        // Layout: bounds[24] = [axis0_min_child0-3][axis0_max_child0-3][axis1_min_child0-3]...
        // That's: [X_min x4][X_max x4][Y_min x4][Y_max x4][Z_min x4][Z_max x4]

        // The decode multiplier is (scale * 65536), so we divide by that
        float invScaleFactor = 1.0f / (g_bvhScale * 65536.0f);

        for (int axis = 0; axis < 3; axis++) {
            for (int child = 0; child < 4; child++) {
                // Convert world coords to encoded int16 values relative to origin
                float minVal = (bounds[child].mins[axis] - g_bvhOrigin[axis]) * invScaleFactor;
                float maxVal = (bounds[child].maxs[axis] - g_bvhOrigin[axis]) * invScaleFactor;

                // Clamp to int16 range (conservative: floor for min, ceil for max)
                minVal = std::clamp(minVal, -32768.0f, 32767.0f);
                maxVal = std::clamp(maxVal, -32768.0f, 32767.0f);

                outBounds[axis * 8 + child] = static_cast<int16_t>(std::floor(minVal));
                outBounds[axis * 8 + 4 + child] = static_cast<int16_t>(std::ceil(maxVal));
            }
        }
    }

    /*
        EmitPoly3Leaf
        Emits a Type 5 (Poly3) triangle leaf using packed vertices
        
        From IDA (CollPoly_Visit_5):
        - Header (4 bytes): 
          - uint16_t [0]: bits 0-11 = surfPropIdx, bits 12-15 = (numPolys - 1)
          - uint16_t [1]: baseVertex (NOT shifted - game does baseVertex << 10)
        - Per-triangle uint32_t:
          - bits 0-10: v0 offset from running base
          - bits 11-19: v1 delta = (v1 - v0 - 1)
          - bits 20-28: v2 delta = (v2 - v0 - 1)
          - bits 29-31: flags (usually 0)
        
        The game's vertex decode:
          running_base starts as (baseVertex << 10)
          v0_global = running_base + v0_offset
          v1_global = v0_global + 1 + v1_delta
          v2_global = v0_global + 1 + v2_delta
          running_base = v0_global (for next triangle)
    */
    int EmitPoly3Leaf(const std::vector<int>& triIndices, int surfPropIdx = 0) {
        if (triIndices.empty()) {
            return ApexLegends::EmitBVHDataleaf();
        }
        
        int numTris = std::min((int)triIndices.size(), 16);  // Max 16 tris (4-bit count)
        int leafIndex = ApexLegends::Bsp::bvhLeafDatas.size();

        // First, emit all vertices for this leaf sequentially
        // The baseVertex will point to where we start adding vertices
        uint32_t baseVertexGlobal = static_cast<uint32_t>(ApexLegends::Bsp::packedVertices.size());
        
        // Emit vertices - 3 per triangle, sequentially
        for (int i = 0; i < numTris; i++) {
            const CollisionTri_t& tri = g_collisionTris[triIndices[i]];
            EmitPackedVertex(tri.v0);
            EmitPackedVertex(tri.v1);
            EmitPackedVertex(tri.v2);
        }
        
        // Calculate model-relative base vertex index
        // The game will compute: running_base = baseVertex << 10
        // We need baseVertex such that (baseVertex << 10) + our_offsets = actual indices
        
        // Since we're laying out vertices sequentially starting at baseVertexGlobal,
        // and the game expects (baseVertex << 10) to point to our base:
        // baseVertex = baseVertexGlobal >> 10 (requires alignment!)
        
        // For simplicity, we'll use baseVertex = 0 and encode absolute offsets
        // This works because we control the vertex layout
        uint32_t baseVertexRelative = baseVertexGlobal - g_modelPackedVertexBase;
        
        // The baseVertex stored must satisfy: (baseVertex << 10) <= baseVertexRelative
        // We'll use the aligned-down value
        uint32_t baseVertexEncoded = baseVertexRelative >> 10;
        uint32_t baseOffset = baseVertexRelative - (baseVertexEncoded << 10);
        
        // Header: bits 0-11 = surfPropIdx, bits 12-15 = numPolys-1
        uint32_t header = (surfPropIdx & 0xFFF) | (((numTris - 1) & 0xF) << 12);
        // Combine header uint16 + baseVertex uint16 into one uint32
        uint32_t headerWord = header | (baseVertexEncoded << 16);
        ApexLegends::Bsp::bvhLeafDatas.push_back(static_cast<int32_t>(headerWord));
        
        // Emit per-triangle vertex index data
        // running_base starts at (baseVertexEncoded << 10) = baseOffset below our first vertex
        uint32_t running_base = baseVertexEncoded << 10;
        
        for (int i = 0; i < numTris; i++) {
            // Our vertex layout: tri i uses vertices at baseVertexRelative + i*3, +i*3+1, +i*3+2
            uint32_t v0_global = baseVertexRelative + i * 3;
            uint32_t v1_global = v0_global + 1;
            uint32_t v2_global = v0_global + 2;

            // v0_offset: v0_global = running_base + v0_offset
            uint32_t v0_offset = v0_global - running_base;
            
            // v1_delta: v1_global = v0_global + 1 + v1_delta
            uint32_t v1_delta = v1_global - (v0_global + 1);  // Should be 0
            
            // v2_delta: v2_global = v0_global + 1 + v2_delta  
            uint32_t v2_delta = v2_global - (v0_global + 1);  // Should be 1
            
            // Pack: bits 0-10 = v0_offset, bits 11-19 = v1_delta, bits 20-28 = v2_delta
            uint32_t triData = (v0_offset & 0x7FF) | ((v1_delta & 0x1FF) << 11) | ((v2_delta & 0x1FF) << 20);
            ApexLegends::Bsp::bvhLeafDatas.push_back(static_cast<int32_t>(triData));
            
            // Update running_base to v0 for next triangle
            running_base = v0_global;
        }
        
        return leafIndex;
    }

    /*
        EmitTriangleStripLeaf  
        Emits a Type 4 (TriStrip) leaf using FLOAT vertices
        
        Type 4 uses full float3 vertices (12 bytes each) instead of packed int16x3.
        The vertex array is at bvh->verts as float3*, not CollPackedPos_s*.
        
        From IDA (CollBvh_VisitLeafs case 4):
        - Uses vertData[12 * idx] - 12-byte stride = float3
        - Header similar to Type 5 but different vertex fetch
        
        For now, we always use Type 5 (Poly3) with packed vertices since
        we set bvhFlags=1 which indicates packed vertex usage.
    */
    int EmitTriangleStripLeaf(const std::vector<int>& triIndices, int surfPropIdx = 0) {
        // Type 4 uses float vertices, but we use packed vertices (bvhFlags=1)
        // So we emit as Type 5 (Poly3) instead, which uses the same format
        // but with packed int16x3 vertices
        return EmitPoly3Leaf(triIndices, surfPropIdx);
    }

    /*
        EmitPoly4Leaf
        Emits a quad polygon leaf (type 6)
        
        Similar format to Type 5, but each polygon references 4 vertices.
        Since we only emit triangles from mesh data, we use EmitPoly3Leaf.
    */
    int EmitPoly4Leaf(const std::vector<int>& quadIndices, int surfPropIdx = 0) {
        // We don't emit quads - convert to triangles and use Poly3
        return EmitPoly3Leaf(quadIndices, surfPropIdx);
    }

    /*
        EmitConvexHullLeaf
        Emits a convex hull leaf (type 8)
        
        From IDA (CollBvh_VisitLeafs case 8):
        - Header bytes [0-3]: numVerts, numFaces, numTris, numPoly5
        - Bytes [4-19]: origin.x, origin.y, origin.z, scale (float4)
        - Then: packed vertices (6 bytes each)
        - Then: face indices
        - Then: embedded Type 5 polys for numTris triangles
        - Then: embedded Type 7 polys for numPoly5 5+ polys
        
        For simplicity, we convert convex hulls to Type 5 triangle leaves.
    */
    int EmitConvexHullLeaf(const std::vector<int>& triIndices, int contentsMaskIdx = 0) {
        // For now, emit as Poly3 (simpler and works for all cases)
        return EmitPoly3Leaf(triIndices, 0);
    }

    /*
        EmitStaticPropLeaf
        Emits a static prop collision reference (type 9)
    */
    int EmitStaticPropLeaf(uint32_t propIndex) {
        int leafIndex = ApexLegends::Bsp::bvhLeafDatas.size();
        ApexLegends::Bsp::bvhLeafDatas.push_back(static_cast<int32_t>(propIndex));
        return leafIndex;
    }

    /*
        EmitHeightfieldLeaf
        Emits a heightfield terrain patch (type 10)
    */
    int EmitHeightfieldLeaf(const CollisionHeightfield_t& hfield) {
        int leafIndex = ApexLegends::Bsp::bvhLeafDatas.size();
        
        // Pack cell position and flags
        uint32_t cellData = hfield.cellX | (hfield.cellY << 8) | (0 << 16);  // flags in upper bits
        ApexLegends::Bsp::bvhLeafDatas.push_back(static_cast<int32_t>(cellData));
        
        // Material indices (4 bytes as one int32)
        uint32_t matData = hfield.materialIndex;  // Simplified - just use one material
        ApexLegends::Bsp::bvhLeafDatas.push_back(static_cast<int32_t>(matData));
        
        // Height samples - 16 int16 values = 8 int32 values
        for (int i = 0; i < 16; i += 2) {
            uint32_t packed = (static_cast<uint16_t>(hfield.heights[i])) |
                             (static_cast<uint16_t>(hfield.heights[i + 1]) << 16);
            ApexLegends::Bsp::bvhLeafDatas.push_back(static_cast<int32_t>(packed));
        }
        
        return leafIndex;
    }

    /*
        EmitBundleLeaf
        Emits a bundle containing multiple child leaf references (type 3)
    */
    int EmitBundleLeaf(const std::vector<std::pair<int, int>>& children) {
        // children is a list of (leafDataOffset, childType) pairs
        if (children.empty()) {
            return ApexLegends::EmitBVHDataleaf();
        }
        
        int leafIndex = ApexLegends::Bsp::bvhLeafDatas.size();
        
        // Bundle header: count
        ApexLegends::Bsp::bvhLeafDatas.push_back(static_cast<int32_t>(children.size()));
        
        // Each child entry: contentsMaskIdx (8 bits), childType (8 bits), leafDataSize (16 bits)
        for (const auto& child : children) {
            uint32_t entry = (0) | (child.second << 8) | (1 << 16);  // contentsMask=0, size=1
            ApexLegends::Bsp::bvhLeafDatas.push_back(static_cast<int32_t>(entry));
        }
        
        return leafIndex;
    }

    /*
        SelectBestLeafType
        Chooses the optimal leaf type for a set of triangles
        Returns the appropriate BVH4_TYPE_* constant
        
        Since we use packed vertices (bvhFlags=1), we MUST use Type 5 (POLY3).
        Type 4 (TRISTRIP) uses float vertices which requires bvhFlags=0.
    */
    int SelectBestLeafType(const std::vector<int>& triIndices) {
        if (triIndices.empty()) {
            return BVH4_TYPE_EMPTY;
        }
        
        // Always use POLY3 (type 5) since we set bvhFlags=1 (packed vertices)
        // Type 4 would require float vertices (bvhFlags=0)
        return BVH4_TYPE_POLY3;
    }

    /*
        EmitLeafDataForType
        Emits leaf data based on the specified type
        Returns index into bvhLeafDatas
        
        Note: We only support Type 5 (Poly3) for now since we use packed vertices.
    */
    int EmitLeafDataForType(int leafType, const std::vector<int>& triIndices, int surfPropIdx = 0) {
        if (triIndices.empty()) {
            return 0;  // Empty leaf
        }
        
        // All types emit as Poly3 since we use packed vertices (bvhFlags=1)
        return EmitPoly3Leaf(triIndices, surfPropIdx);
    }

    /*
        ComputeTriangleBounds
        Computes AABB for a triangle
    */
    MinMax ComputeTriangleBounds(const CollisionTri_t& tri) {
        MinMax bounds;
        bounds.mins = Vec3Min(Vec3Min(tri.v0, tri.v1), tri.v2);
        bounds.maxs = Vec3Max(Vec3Max(tri.v0, tri.v1), tri.v2);
        return bounds;
    }

    /*
        ComputeTriangleArea
        Computes the area of a triangle using cross product
    */
    float ComputeTriangleArea(const CollisionTri_t& tri) {
        Vector3 edge1 = tri.v1 - tri.v0;
        Vector3 edge2 = tri.v2 - tri.v0;
        Vector3 cross = vector3_cross(edge1, edge2);
        return vector3_length(cross) * 0.5f;
    }

    /*
        ComputeMinEdgeLength
        Returns the length of the shortest edge in a triangle
    */
    float ComputeMinEdgeLength(const CollisionTri_t& tri) {
        float edge0 = vector3_length(tri.v1 - tri.v0);
        float edge1 = vector3_length(tri.v2 - tri.v1);
        float edge2 = vector3_length(tri.v0 - tri.v2);
        return std::min({edge0, edge1, edge2});
    }

    /*
        IsDegenerateTriangle
        Checks if a triangle is too small or degenerate
    */
    bool IsDegenerateTriangle(const CollisionTri_t& tri) {
        // Check minimum edge length
        if (ComputeMinEdgeLength(tri) < MIN_TRIANGLE_EDGE) {
            return true;
        }

        // Check minimum area
        if (ComputeTriangleArea(tri) < MIN_TRIANGLE_AREA) {
            return true;
        }

        return false;
    }

    /*
        SnapVertexToGrid
        Snaps a vertex to a fixed grid to eliminate floating point precision issues
        This ensures that vertices that should be at the same position actually are
        Grid size of 0.125 (1/8th unit) provides good precision while preventing cracks
    */
    Vector3 SnapVertexToGrid(const Vector3& vert) {
        constexpr float GRID_SIZE = 0.125f;  // 1/8th unit grid
        float invGrid = 1.0f / GRID_SIZE;

        return Vector3(
            std::round(vert.x() * invGrid) / invGrid,
            std::round(vert.y() * invGrid) / invGrid,
            std::round(vert.z() * invGrid) / invGrid
        );
    }

    /*
        ComputeBoundsForTriangles
        Computes combined AABB for a set of triangles
    */
    MinMax ComputeBoundsForTriangles(const std::vector<int>& triIndices) {
        MinMax bounds;
        bounds.mins = Vector3(std::numeric_limits<float>::max());
        bounds.maxs = Vector3(std::numeric_limits<float>::lowest());
        
        for (int idx : triIndices) {
            const CollisionTri_t& tri = g_collisionTris[idx];
            MinMax triBounds = ComputeTriangleBounds(tri);
            bounds.mins = Vec3Min(bounds.mins, triBounds.mins);
            bounds.maxs = Vec3Max(bounds.maxs, triBounds.maxs);
        }
        
        return bounds;
    }

    /*
        ComputeTriangleCentroid
        Returns the centroid of a triangle
    */
    Vector3 ComputeTriangleCentroid(const CollisionTri_t& tri) {
        return (tri.v0 + tri.v1 + tri.v2) / 3.0f;
    }

    /*
        PartitionTriangles
        Partitions triangles into up to 4 groups for BVH4 using SAH-like heuristic
        Returns number of partitions created (1-4)
    */
    int PartitionTriangles(const std::vector<int>& triIndices, const MinMax& bounds,
                           std::vector<int> outPartitions[4]) {
        if (triIndices.size() <= MAX_TRIS_PER_LEAF) {
            outPartitions[0] = triIndices;
            return 1;
        }

        // Find longest axis
        Vector3 size = bounds.maxs - bounds.mins;
        int axis = 0;
        if (size.y() > size.x()) axis = 1;
        if (size.z() > size[axis]) axis = 2;

        // Sort by centroid along axis
        std::vector<int> sorted = triIndices;
        std::sort(sorted.begin(), sorted.end(), [axis](int a, int b) {
            Vector3 centA = ComputeTriangleCentroid(g_collisionTris[a]);
            Vector3 centB = ComputeTriangleCentroid(g_collisionTris[b]);
            return centA[axis] < centB[axis];
        });

        // Split into 2-4 partitions based on count
        size_t count = sorted.size();
        int numPartitions;
        
        if (count >= 16) {
            numPartitions = 4;
        } else if (count >= 8) {
            numPartitions = 4;
        } else {
            numPartitions = 2;
        }

        // Distribute triangles evenly
        for (int i = 0; i < numPartitions; i++) {
            outPartitions[i].clear();
        }
        
        size_t trisPerPartition = (count + numPartitions - 1) / numPartitions;
        for (size_t i = 0; i < count; i++) {
            int partition = std::min((int)(i / trisPerPartition), numPartitions - 1);
            outPartitions[partition].push_back(sorted[i]);
        }

        // Remove empty partitions
        int actualPartitions = 0;
        for (int i = 0; i < numPartitions; i++) {
            if (!outPartitions[i].empty()) {
                if (i != actualPartitions) {
                    outPartitions[actualPartitions] = std::move(outPartitions[i]);
                    outPartitions[i].clear();
                }
                actualPartitions++;
            }
        }

        return actualPartitions;
    }

    /*
        BuildBVH4Node
        Recursively builds BVH4 tree from triangles
        Returns index of created node in g_bvhBuildNodes
    */
    int BuildBVH4Node(const std::vector<int>& triIndices, int depth) {
        if (triIndices.empty()) {
            return -1;
        }

        int nodeIndex = g_bvhBuildNodes.size();
        g_bvhBuildNodes.emplace_back();
        
        // Use index access instead of reference since vector may reallocate during recursion
        g_bvhBuildNodes[nodeIndex].bounds = ComputeBoundsForTriangles(triIndices);
        
        // Collect content flags from all triangles
        g_bvhBuildNodes[nodeIndex].contentFlags = 0;
        for (int idx : triIndices) {
            g_bvhBuildNodes[nodeIndex].contentFlags |= g_collisionTris[idx].contentFlags;
        }
        if (g_bvhBuildNodes[nodeIndex].contentFlags == 0) {
            g_bvhBuildNodes[nodeIndex].contentFlags = CONTENTS_SOLID;
        }

        // If few triangles or max depth, make leaf
        if (triIndices.size() <= MAX_TRIS_PER_LEAF || depth >= MAX_BVH_DEPTH) {
            g_bvhBuildNodes[nodeIndex].isLeaf = true;
            g_bvhBuildNodes[nodeIndex].triangleIndices = triIndices;
            return nodeIndex;
        }

        // Partition triangles - save bounds locally since we'll need it after potential reallocation
        MinMax nodeBounds = g_bvhBuildNodes[nodeIndex].bounds;
        
        std::vector<int> partitions[4];
        int numPartitions = PartitionTriangles(triIndices, nodeBounds, partitions);

        // If we couldn't partition effectively, make leaf
        if (numPartitions <= 1) {
            g_bvhBuildNodes[nodeIndex].isLeaf = true;
            g_bvhBuildNodes[nodeIndex].triangleIndices = triIndices;
            return nodeIndex;
        }

        // Build child nodes
        g_bvhBuildNodes[nodeIndex].isLeaf = false;
        for (int i = 0; i < 4; i++) {
            if (i < numPartitions && !partitions[i].empty()) {
                // Check if this partition should be a leaf
                if (partitions[i].size() <= MAX_TRIS_PER_LEAF) {
                    // Create leaf node
                    int leafIndex = g_bvhBuildNodes.size();
                    g_bvhBuildNodes.emplace_back();
                    g_bvhBuildNodes[leafIndex].bounds = ComputeBoundsForTriangles(partitions[i]);
                    g_bvhBuildNodes[leafIndex].isLeaf = true;
                    g_bvhBuildNodes[leafIndex].triangleIndices = partitions[i];
                    g_bvhBuildNodes[leafIndex].contentFlags = 0;
                    for (int idx : partitions[i]) {
                        g_bvhBuildNodes[leafIndex].contentFlags |= g_collisionTris[idx].contentFlags;
                    }
                    if (g_bvhBuildNodes[leafIndex].contentFlags == 0) {
                        g_bvhBuildNodes[leafIndex].contentFlags = CONTENTS_SOLID;
                    }
                    g_bvhBuildNodes[nodeIndex].childIndices[i] = leafIndex;
                    g_bvhBuildNodes[nodeIndex].childTypes[i] = SelectBestLeafType(partitions[i]);
                } else {
                    // Recursively build
                    int childIdx = BuildBVH4Node(partitions[i], depth + 1);
                    g_bvhBuildNodes[nodeIndex].childIndices[i] = childIdx;
                    if (childIdx >= 0) {
                        if (g_bvhBuildNodes[childIdx].isLeaf) {
                            g_bvhBuildNodes[nodeIndex].childTypes[i] = SelectBestLeafType(g_bvhBuildNodes[childIdx].triangleIndices);
                        } else {
                            g_bvhBuildNodes[nodeIndex].childTypes[i] = BVH4_TYPE_NODE;
                        }
                    }
                }
            } else {
                g_bvhBuildNodes[nodeIndex].childIndices[i] = -1;
                g_bvhBuildNodes[nodeIndex].childTypes[i] = BVH4_TYPE_NONE;
            }
        }

        return nodeIndex;
    }

    /*
        EmitBVH4Nodes
        Converts built BVH tree to Apex BSP format and emits to bvhNodes
        Returns index of root node in bvhNodes
    */
    int EmitBVH4Nodes(int buildNodeIndex, int& leafDataOffset) {
        if (buildNodeIndex < 0 || buildNodeIndex >= (int)g_bvhBuildNodes.size()) {
            return -1;
        }

        const BVHBuildNode_t& buildNode = g_bvhBuildNodes[buildNodeIndex];
        int bspNodeIndex = ApexLegends::Bsp::bvhNodes.size();
        ApexLegends::Bsp::bvhNodes.emplace_back();
        
        // Initialize with defaults - use index access since vector may reallocate
        memset(&ApexLegends::Bsp::bvhNodes[bspNodeIndex], 0, sizeof(ApexLegends::BVHNode_t));
        ApexLegends::Bsp::bvhNodes[bspNodeIndex].cmIndex = ApexLegends::EmitContentsMask(buildNode.contentFlags);

        if (buildNode.isLeaf) {
            // Leaf node - emit as single child with appropriate type
            int leafType = SelectBestLeafType(buildNode.triangleIndices);
            ApexLegends::Bsp::bvhNodes[bspNodeIndex].childType0 = leafType;
            ApexLegends::Bsp::bvhNodes[bspNodeIndex].childType1 = BVH4_TYPE_NONE;
            ApexLegends::Bsp::bvhNodes[bspNodeIndex].childType2 = BVH4_TYPE_NONE;
            ApexLegends::Bsp::bvhNodes[bspNodeIndex].childType3 = BVH4_TYPE_NONE;
            
            // Emit leaf data based on type
            ApexLegends::Bsp::bvhNodes[bspNodeIndex].index0 = EmitLeafDataForType(leafType, buildNode.triangleIndices);
            ApexLegends::Bsp::bvhNodes[bspNodeIndex].index1 = 0;
            ApexLegends::Bsp::bvhNodes[bspNodeIndex].index2 = 0;
            ApexLegends::Bsp::bvhNodes[bspNodeIndex].index3 = 0;
            
            // Pack bounds for this single child
            MinMax childBounds[4];
            childBounds[0] = buildNode.bounds;
            childBounds[1] = childBounds[2] = childBounds[3] = buildNode.bounds;  // Unused slots
            PackBoundsToInt16(childBounds, ApexLegends::Bsp::bvhNodes[bspNodeIndex].bounds);
            
        } else {
            // Internal node - emit all children
            MinMax childBounds[4];
            
            for (int i = 0; i < 4; i++) {
                if (buildNode.childIndices[i] >= 0) {
                    const BVHBuildNode_t& childBuild = g_bvhBuildNodes[buildNode.childIndices[i]];
                    childBounds[i] = childBuild.bounds;
                } else {
                    // Empty slot - use parent bounds (will be ignored due to NONE type)
                    childBounds[i] = buildNode.bounds;
                }
            }
            
            PackBoundsToInt16(childBounds, ApexLegends::Bsp::bvhNodes[bspNodeIndex].bounds);
            
            // Set child types
            ApexLegends::Bsp::bvhNodes[bspNodeIndex].childType0 = buildNode.childTypes[0];
            ApexLegends::Bsp::bvhNodes[bspNodeIndex].childType1 = buildNode.childTypes[1];
            ApexLegends::Bsp::bvhNodes[bspNodeIndex].childType2 = buildNode.childTypes[2];
            ApexLegends::Bsp::bvhNodes[bspNodeIndex].childType3 = buildNode.childTypes[3];
            
            // Recursively emit children and get their indices
            for (int i = 0; i < 4; i++) {
                int childIndex = 0;
                int childType = buildNode.childTypes[i];
                
                if (buildNode.childIndices[i] >= 0) {
                    if (childType == BVH4_TYPE_NODE) {
                        // Recurse for internal nodes
                        childIndex = EmitBVH4Nodes(buildNode.childIndices[i], leafDataOffset);
                    } else if (childType != BVH4_TYPE_NONE && childType != BVH4_TYPE_EMPTY) {
                        // Emit leaf data based on type
                        const BVHBuildNode_t& childBuild = g_bvhBuildNodes[buildNode.childIndices[i]];
                        childIndex = EmitLeafDataForType(childType, childBuild.triangleIndices);
                    }
                }
                
                // Access via index since vector may have reallocated during recursion
                switch (i) {
                    case 0: ApexLegends::Bsp::bvhNodes[bspNodeIndex].index0 = childIndex; break;
                    case 1: ApexLegends::Bsp::bvhNodes[bspNodeIndex].index1 = childIndex; break;
                    case 2: ApexLegends::Bsp::bvhNodes[bspNodeIndex].index2 = childIndex; break;
                    case 3: ApexLegends::Bsp::bvhNodes[bspNodeIndex].index3 = childIndex; break;
                }
            }
        }

        return bspNodeIndex;
    }

    /*
        CollectTrianglesFromMeshes
        Collects collision triangles from Shared::meshes
        Filters out degenerate triangles
    */
    void CollectTrianglesFromMeshes() {
        g_collisionTris.clear();

        int skippedDegenerate = 0;
        int totalTris = 0;

        for (const Shared::Mesh_t& mesh : Shared::meshes) {
            // Get content flags from shader
            int contentFlags = CONTENTS_SOLID;
            if (mesh.shaderInfo) {
                contentFlags = mesh.shaderInfo->contentFlags;

                // Skip non-solid surfaces
                if (!(contentFlags & CONTENTS_SOLID) &&
                    !(contentFlags & CONTENTS_PLAYERCLIP) &&
                    !(contentFlags & CONTENTS_MONSTERCLIP)) {
                    // Non-solid content - skip collision
                    continue;
                }
            }

            // Extract triangles
            const std::vector<Shared::Vertex_t>& verts = mesh.vertices;
            const std::vector<uint16_t>& indices = mesh.triangles;

            for (size_t i = 0; i + 2 < indices.size(); i += 3) {
                totalTris++;

                CollisionTri_t tri;
                tri.v0 = verts[indices[i]].xyz;
                tri.v1 = verts[indices[i + 1]].xyz;
                tri.v2 = verts[indices[i + 2]].xyz;

                // Compute normal
                Vector3 edge1 = tri.v1 - tri.v0;
                Vector3 edge2 = tri.v2 - tri.v0;
                tri.normal = vector3_cross(edge1, edge2);
                float len = vector3_length(tri.normal);
                if (len > 0.0001f) {
                    tri.normal = tri.normal / len;
                }

                // Skip degenerate triangles (too small or sliver triangles)
                if (IsDegenerateTriangle(tri)) {
                    skippedDegenerate++;
                    continue;
                }

                tri.contentFlags = contentFlags;
                tri.surfaceFlags = mesh.shaderInfo ? mesh.shaderInfo->surfaceFlags : 0;

                g_collisionTris.push_back(tri);
            }
        }

        if (skippedDegenerate > 0) {
            Sys_FPrintf(SYS_VRB, "  Filtered %d degenerate triangles (%.1f%%)\n",
                       skippedDegenerate, 100.0f * skippedDegenerate / totalTris);
        }
        Sys_Printf("  Collected %zu collision triangles (from %d total)\n",
                   g_collisionTris.size(), totalTris);
    }

}  // anonymous namespace


/*
    EmitBVHNode
    Builds BVH4 collision tree for current model from mesh triangles
*/
void ApexLegends::EmitBVHNode() {
    Sys_FPrintf(SYS_VRB, "Building BVH4 collision tree...\n");
    
    // Get the current model to update
    ApexLegends::Model_t& model = ApexLegends::Bsp::models.back();
    
    // Record starting indices for this model's BVH data
    model.bvhNodeIndex = ApexLegends::Bsp::bvhNodes.size();
    model.bvhLeafIndex = ApexLegends::Bsp::bvhLeafDatas.size();
    
    // Collect triangles from meshes
    CollectTrianglesFromMeshes();
    
    if (g_collisionTris.empty()) {
        // No collision geometry - emit a single empty node
        Sys_FPrintf(SYS_WRN, "Warning: No collision triangles, emitting empty BVH node\n");
        
        // Set default origin/scale for empty models
        model.origin[0] = model.origin[1] = model.origin[2] = 0.0f;
        model.scale = 1.0f / 65536.0f;
        model.vertexIndex = 0;
        model.bvhFlags = 0;
        
        ApexLegends::BVHNode_t& node = ApexLegends::Bsp::bvhNodes.emplace_back();
        memset(&node, 0, sizeof(node));
        node.cmIndex = EmitContentsMask(CONTENTS_SOLID);
        node.childType0 = BVH4_TYPE_NONE;
        node.childType1 = BVH4_TYPE_NONE;
        node.childType2 = BVH4_TYPE_NONE;
        node.childType3 = BVH4_TYPE_NONE;
        return;
    }
    
    // Calculate overall bounds from collision triangles
    MinMax overallBounds;
    overallBounds.mins = Vector3(std::numeric_limits<float>::max());
    overallBounds.maxs = Vector3(std::numeric_limits<float>::lowest());
    
    for (const CollisionTri_t& tri : g_collisionTris) {
        overallBounds.mins = Vec3Min(overallBounds.mins, Vec3Min(Vec3Min(tri.v0, tri.v1), tri.v2));
        overallBounds.maxs = Vec3Max(overallBounds.maxs, Vec3Max(Vec3Max(tri.v0, tri.v1), tri.v2));
    }
    
    // Calculate origin (center of bounds) and scale
    Vector3 center = (overallBounds.mins + overallBounds.maxs) * 0.5f;
    Vector3 extents = (overallBounds.maxs - overallBounds.mins) * 0.5f;
    float maxExtent = std::max({extents.x(), extents.y(), extents.z(), 1.0f});
    
    // The game's decode formula is: worldPos = origin + (int16 << 16) * scale
    // Which equals: worldPos = origin + int16 * 65536 * scale
    // 
    // For simplicity, use scale = 1/65536 so that int16 directly maps to world units:
    //   worldPos = origin + int16 * 65536 * (1/65536) = origin + int16
    //
    // This gives a range of ±32767 units from origin.
    // If geometry exceeds this, we need a larger scale.
    float bvhScale;
    if (maxExtent <= 32000.0f) {
        // Use 1:1 mapping
        bvhScale = 1.0f / 65536.0f;
    } else {
        // Scale up to fit larger geometry
        bvhScale = maxExtent / (32000.0f * 65536.0f);
    }
    
    // Store in global for PackBoundsToInt16 to use
    g_bvhOrigin = center;
    g_bvhScale = bvhScale;
    
    // Record base vertex index for this model's collision vertices
    g_modelPackedVertexBase = static_cast<uint32_t>(ApexLegends::Bsp::packedVertices.size());
    
    // Store in model
    model.origin[0] = center.x();
    model.origin[1] = center.y();
    model.origin[2] = center.z();
    model.scale = bvhScale;
    model.vertexIndex = g_modelPackedVertexBase;  // Index into packed vertices for this model
    model.bvhFlags = 1;  // Use packed vertices (int16x3)
    
    Sys_FPrintf(SYS_VRB, "  BVH origin: (%.1f, %.1f, %.1f), scale: %g\n", 
               center.x(), center.y(), center.z(), bvhScale);
    
    // Build triangle index list
    std::vector<int> allTriIndices(g_collisionTris.size());
    std::iota(allTriIndices.begin(), allTriIndices.end(), 0);
    
    // Build BVH tree
    g_bvhBuildNodes.clear();
    int rootBuildIndex = BuildBVH4Node(allTriIndices, 0);
    
    if (rootBuildIndex < 0) {
        Sys_FPrintf(SYS_WRN, "Warning: BVH build failed, emitting empty node\n");
        ApexLegends::BVHNode_t& node = ApexLegends::Bsp::bvhNodes.emplace_back();
        memset(&node, 0, sizeof(node));
        node.cmIndex = EmitContentsMask(CONTENTS_SOLID);
        node.childType0 = BVH4_TYPE_NONE;
        node.childType1 = BVH4_TYPE_NONE;
        node.childType2 = BVH4_TYPE_NONE;
        node.childType3 = BVH4_TYPE_NONE;
        return;
    }
    
    // Emit to BSP format
    int leafDataOffset = 0;
    int rootNodeIndex = EmitBVH4Nodes(rootBuildIndex, leafDataOffset);
    (void)rootNodeIndex;  // Unused but documented
    
    Sys_FPrintf(SYS_VRB, "  Emitted %zu BVH nodes, %zu leaf data entries, %zu packed vertices\n", 
               ApexLegends::Bsp::bvhNodes.size() - model.bvhNodeIndex,
               ApexLegends::Bsp::bvhLeafDatas.size() - model.bvhLeafIndex,
               ApexLegends::Bsp::packedVertices.size() - g_modelPackedVertexBase);
    
    // Clean up build data
    g_bvhBuildNodes.clear();
    g_collisionTris.clear();
}

/*
    EmitBVHDataleaf
    Emits a leaf data entry for collision
    Returns index into bvhLeafDatas
*/
int ApexLegends::EmitBVHDataleaf() {
    // For now, emit a placeholder leaf data value
    // The actual format depends on the child type:
    // - ConvexHull: brush index or convex hull data
    // - TriStrip/Poly: triangle strip data
    // The engine uses this as an index into collision mesh data
    
    int index = ApexLegends::Bsp::bvhLeafDatas.size();
    ApexLegends::Bsp::bvhLeafDatas.emplace_back(0);  // Placeholder value
    return index;
}

/*
    EmitContentsMask
    Emits collision flags and returns an index to them
*/
int ApexLegends::EmitContentsMask(int mask) {
    for (size_t i = 0; i < ApexLegends::Bsp::contentsMasks.size(); i++) {
        if (ApexLegends::Bsp::contentsMasks[i] == mask) {
            return static_cast<int>(i);
        }
    }

    // Didn't find our mask, make new one
    ApexLegends::Bsp::contentsMasks.emplace_back(mask);
    return static_cast<int>(ApexLegends::Bsp::contentsMasks.size() - 1);
}