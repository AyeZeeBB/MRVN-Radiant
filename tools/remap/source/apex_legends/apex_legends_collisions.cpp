/* -------------------------------------------------------------------------------

   Copyright (C) 2022-2023 MRVN-Radiant and contributors.
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
#include <ctime>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cstring>

/*
    BVH4 Collision System for Apex Legends BSP
    
    Based on reverse engineering of Apex Legends collision structures:
    - CollBvh4Node_s: 64-byte BVH4 nodes with packed bounds and child info
    - Child types: 0=Node, 1=None, 2=Empty, 3=Bundle, 4=TriStrip, 5/6/7=Poly, 8=ConvexHull, 9=StaticProp, 10=Heightfield
    - Bounds are stored as int16_t values, quantized from floats
    
    Leaf data formats from IDA reverse engineering:
    
    Type 3 (Bundle): Contains count + array of (contentsMaskIdx, childType, leafDataSize) entries
    Type 4 (TriStrip): Packed triangle strip with baseVertex and encoded vertex deltas
    Type 5 (Poly3): Triangle polygon using CollLeafPoly_s format
    Type 6 (Poly4): Quad polygon using CollLeafPoly_s format  
    Type 7 (Poly5+): 5+ vertex polygon using CollLeafPoly_s format
    Type 8 (ConvexHull): CollLeafConvexHull_s with origin, scale, verts, faces, planes
    Type 9 (StaticProp): Just a uint32_t prop index
    Type 10 (Heightfield): CollLeafHeightfieldPatch_s with height samples and material info
*/

namespace {
    // BVH4 child types from IDA analysis of CollBvh_VisitLeafs
    constexpr int BVH4_TYPE_NODE        = 0;
    constexpr int BVH4_TYPE_NONE        = 1;
    constexpr int BVH4_TYPE_EMPTY       = 2;
    constexpr int BVH4_TYPE_BUNDLE      = 3;
    constexpr int BVH4_TYPE_TRISTRIP    = 4;
    constexpr int BVH4_TYPE_POLY3       = 5;
    constexpr int BVH4_TYPE_POLY4       = 6;
    constexpr int BVH4_TYPE_POLY5PLUS   = 7;
    constexpr int BVH4_TYPE_CONVEXHULL  = 8;
    constexpr int BVH4_TYPE_STATICPROP  = 9;
    constexpr int BVH4_TYPE_HEIGHTFIELD = 10;

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

    // Vertex welding epsilon for snapping close vertices
    // Increased from 0.05 to 0.25 to better handle corner cases
    constexpr float VERTEX_WELD_EPSILON = 0.25f;

    /*
        CollLeafPoly_s - Polygon leaf structure (from IDA)
        Used for types 5, 6, 7 (triangles, quads, and larger polys)
        
        Layout:
        - uint16_t numPolysAndSurfPropIdxAndFlags: bits 0-11 = surfPropIdx, bits 12-15 = numPolys-1
        - uint16_t baseVertex: base vertex index << 10
        - uint32_t polys[]: packed vertex indices per polygon
          Each poly entry: bits 0-10 = v0 offset, bits 11-19 = v1 delta, bits 20-28 = v2 delta, bits 29-31 = surfPropIdx
    */
    struct CollLeafPoly_s {
        uint16_t numPolysAndSurfPropIdxAndFlags;
        uint16_t baseVertex;
        uint32_t polys[1];  // Variable length array
    };

    /*
        CollLeafConvexHull_s - Convex hull collision (from IDA)
        68 bytes total:
        - uint8_t numVerts
        - uint8_t numFaces
        - uint8_t numTris
        - uint8_t numPoly5
        - float4 origin (with scale in w)
        - CollPackedPos_s verts[] (6 bytes each: int16_t x, y, z)
        - uint8_t faceVertexIndices[] (3 per face, packed)
        - CollLeafConvexHullPlane_s planes[] (3 bytes each)
    */
    struct CollLeafConvexHull_s {
        uint8_t numVerts;
        uint8_t numFaces;
        uint8_t numTris;      // Number of triangle faces
        uint8_t numPoly5;     // Number of 5+ vertex faces
        float origin[4];       // x, y, z, scale
        // Followed by: packed verts, face indices, planes
    };

    /*
        CollLeafStaticProp_s - Static prop reference (from IDA)
        4 bytes: just a prop index
    */
    struct CollLeafStaticProp_s {
        uint32_t propIndex;
    };

    /*
        CollLeafHeightfieldPatch_s - Heightfield terrain patch (from IDA)
        40 bytes:
        - uint8_t cellX, cellY - patch position in grid
        - uint16_t flags
        - uint8_t materialIndices[4] - up to 4 material layers
        - uint8_t heightfieldIndex - index into CollBvh4_s::heightfields
        - uint8_t reserved
        - int16_t heights[16] - 4x4 grid of height samples
    */
    struct CollLeafHeightfieldPatch_s {
        uint8_t cellX;
        uint8_t cellY;
        uint16_t flags;
        uint8_t materialIndices[4];
        uint8_t heightfieldIndex;
        uint8_t reserved;
        int16_t heights[16];  // 4x4 grid
    };

    /*
        CollLeafTriStrip_s - Triangle strip data (from IDA)
        Variable size:
        - uint16_t header: bits 0-11 = surfPropIdx, bits 12-15 = numTris-1
        - uint16_t baseVertex: base vertex << 10
        - uint32_t strips[]: packed vertex delta triplets per triangle
          Each strip: bits 0-10 = v0, bits 11-19 = v1 delta, bits 20-28 = v2 delta, bits 29-31 = flags
    */
    struct CollLeafTriStrip_s {
        uint16_t header;
        uint16_t baseVertex;
        uint32_t strips[1];  // Variable length
    };

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

        // Debug: print first vertex to verify encoding
        if (idx == g_modelPackedVertexBase) {
            Sys_Printf("    First packed vertex: world(%.1f,%.1f,%.1f) -> int16(%d,%d,%d)\n",
                       worldPos.x(), worldPos.y(), worldPos.z(),
                       vert.x, vert.y, vert.z);
            Sys_Printf("      invScaleFactor=%.6f, decoded=(%.1f,%.1f,%.1f)\n",
                       invScaleFactor,
                       g_bvhOrigin.x() + vert.x * 65536.0f * g_bvhScale,
                       g_bvhOrigin.y() + vert.y * 65536.0f * g_bvhScale,
                       g_bvhOrigin.z() + vert.z * 65536.0f * g_bvhScale);
        }

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
        EmitTriangleStripLeaf
        Emits a triangle strip leaf (type 4)
        
        From IDA reverse engineering of Coll_Broad_PointVsTriangles:
        - Header uint32: bits 0-11 = surfPropIdx, bits 12-15 = numTris-1, bits 16-31 = baseVertex >> 10
        - Per-triangle uint32: bits 0-10 = v0 offset from running_base, bits 11-19 = v1 delta from v0+1, bits 20-28 = v2 delta from v0+1
        
        The game computes vertices as (with running_base updated each iteration):
        - running_base starts as (baseVertex << 10)
        - v0 = running_base + v0_offset
        - v1 = v0 + 1 + v1_delta
        - v2 = v0 + 1 + v2_delta
        - running_base = v0 (for next triangle)
    */
    int EmitTriangleStripLeaf(const std::vector<int>& triIndices, int surfPropIdx = 0) {
        if (triIndices.empty()) {
            return ApexLegends::EmitBVHDataleaf();
        }
        
        int numTris = std::min((int)triIndices.size(), 16);  // Max 16 tris (4-bit count)

        int leafIndex = ApexLegends::Bsp::bvhLeafDatas.size();

        // The game decodes baseVertex as: running_base = baseVertex << 10
        // So baseVertex must be aligned to 1024-vertex boundaries
        // Pad the vertex array to ensure alignment
        uint32_t baseVertexGlobal = static_cast<uint32_t>(ApexLegends::Bsp::packedVertices.size());

        // Calculate how many vertices we need to add to align to 1024 boundary
        uint32_t alignmentPadding = (1024 - (baseVertexGlobal % 1024)) % 1024;

        // Add padding vertices if needed
        for (uint32_t i = 0; i < alignmentPadding; i++) {
            ApexLegends::PackedVertex_t padVert = {0, 0, 0};
            ApexLegends::Bsp::packedVertices.push_back(padVert);
        }

        // Now baseVertexGlobal is aligned to 1024
        baseVertexGlobal = static_cast<uint32_t>(ApexLegends::Bsp::packedVertices.size());

        // Convert to model-relative index
        uint32_t baseVertexRelative = baseVertexGlobal - g_modelPackedVertexBase;
        
        Sys_Printf("    EmitTriangleStripLeaf: %d tris, baseVertexGlobal=%d, baseVertexRelative=%d\n",
                   numTris, baseVertexGlobal, baseVertexRelative);
        
        for (int i = 0; i < numTris; i++) {
            const CollisionTri_t& tri = g_collisionTris[triIndices[i]];
            uint32_t idx0 = EmitPackedVertex(tri.v0);
            uint32_t idx1 = EmitPackedVertex(tri.v1);
            uint32_t idx2 = EmitPackedVertex(tri.v2);
            Sys_Printf("      Tri %d: verts at global indices [%d, %d, %d]\n", i, idx0, idx1, idx2);
        }
        
        // Header: bits 0-11 = surfPropIdx, bits 12-15 = numTris-1, bits 16-31 = baseVertex >> 10
        // Since baseVertexRelative is aligned to 1024, we encode it as baseVertexRelative >> 10
        // The game will decode as: running_base = (baseVertexRelative >> 10) << 10 = baseVertexRelative
        uint32_t baseVertexEncoded = (baseVertexRelative >> 10) & 0xFFFF;
        uint32_t header = (surfPropIdx & 0xFFF) | (((numTris - 1) & 0xF) << 12) | (baseVertexEncoded << 16);
        ApexLegends::Bsp::bvhLeafDatas.push_back(static_cast<int32_t>(header));
        
        Sys_Printf("      Header: 0x%08X (surfProp=%d, numTris=%d, baseVertexEncoded=%d)\n",
                   header, surfPropIdx, numTris, baseVertexEncoded);
        
        // Emit per-triangle data
        // The game uses a running_base that starts as (baseVertexEncoded << 10) and updates to v0 each iteration
        // For triangle i, our v0 is at baseVertexEncoded + i*3

        uint32_t running_base = baseVertexEncoded << 10;  // Initial running base

        for (int i = 0; i < numTris; i++) {
            // Our vertex layout: base + i*3 = v0, base + i*3 + 1 = v1, base + i*3 + 2 = v2
            uint32_t v0_index = (baseVertexEncoded << 10) + i * 3;
            uint32_t v1_index = v0_index + 1;
            uint32_t v2_index = v0_index + 2;

            // v0_offset: v0 = running_base + v0_offset, so v0_offset = v0_index - running_base
            uint32_t v0_offset = v0_index - running_base;
            
            // v1 = v0 + 1 + v1_delta, so v1_delta = v1_index - (v0_index + 1) = 0
            uint32_t v1_delta = v1_index - (v0_index + 1);  // Should be 0
            
            // v2 = v0 + 1 + v2_delta, so v2_delta = v2_index - (v0_index + 1) = 1
            uint32_t v2_delta = v2_index - (v0_index + 1);  // Should be 1
            
            // Pack: bits 0-10 = v0_offset, bits 11-19 = v1_delta, bits 20-28 = v2_delta
            uint32_t triData = (v0_offset & 0x7FF) | ((v1_delta & 0x1FF) << 11) | ((v2_delta & 0x1FF) << 20);
            ApexLegends::Bsp::bvhLeafDatas.push_back(static_cast<int32_t>(triData));
            
            Sys_Printf("      Tri %d: v0_idx=%d, running_base=%d, v0_offset=%d, triData=0x%08X\n",
                       i, v0_index, running_base, v0_offset, triData);
            Sys_Printf("        Game will decode: v0=%d, v1=%d, v2=%d\n",
                       running_base + v0_offset, running_base + v0_offset + 1 + v1_delta, 
                       running_base + v0_offset + 1 + v2_delta);
            
            // Update running_base to v0 for next triangle
            running_base = v0_index;
        }
        
        return leafIndex;
    }

    /*
        EmitPoly3Leaf
        Emits a triangle polygon leaf (type 5)
        Uses same format as EmitTriangleStripLeaf - both use CollLeafPoly_s format
    */
    int EmitPoly3Leaf(const std::vector<int>& triIndices, int surfPropIdx = 0) {
        // Poly3 and TriStrip use the same format, just different child types
        return EmitTriangleStripLeaf(triIndices, surfPropIdx);
    }

    /*
        EmitPoly4Leaf
        Emits a quad polygon leaf (type 6)
    */
    int EmitPoly4Leaf(const std::vector<int>& quadIndices, int surfPropIdx = 0) {
        // Similar to Poly3 but for quads (4 vertices per poly)
        if (quadIndices.empty()) {
            return ApexLegends::EmitBVHDataleaf();
        }
        
        int leafIndex = ApexLegends::Bsp::bvhLeafDatas.size();
        
        // For quads, we need 4 vertex references per polygon
        // Header format is the same
        uint32_t header = (surfPropIdx & 0xFFF) | (((1 - 1) & 0xF) << 12);  // 1 quad
        ApexLegends::Bsp::bvhLeafDatas.push_back(static_cast<int32_t>(header));
        
        // Quad data - 4 vertex indices packed
        ApexLegends::Bsp::bvhLeafDatas.push_back(0);  // Placeholder
        
        return leafIndex;
    }

    /*
        EmitConvexHullLeaf
        Emits a convex hull leaf (type 8)
        This is the primary type for brush collision
    */
    int EmitConvexHullLeaf(const std::vector<int>& triIndices, int contentsMaskIdx = 0) {
        if (triIndices.empty()) {
            return ApexLegends::EmitBVHDataleaf();
        }
        
        int leafIndex = ApexLegends::Bsp::bvhLeafDatas.size();
        
        // Compute bounds for origin/scale calculation
        MinMax bounds;
        bounds.mins = Vector3(std::numeric_limits<float>::max());
        bounds.maxs = Vector3(std::numeric_limits<float>::lowest());
        
        for (int idx : triIndices) {
            const CollisionTri_t& tri = g_collisionTris[idx];
            bounds.mins = Vec3Min(bounds.mins, Vec3Min(Vec3Min(tri.v0, tri.v1), tri.v2));
            bounds.maxs = Vec3Max(bounds.maxs, Vec3Max(Vec3Max(tri.v0, tri.v1), tri.v2));
        }
        
        Vector3 center = (bounds.mins + bounds.maxs) * 0.5f;
        Vector3 extents = (bounds.maxs - bounds.mins) * 0.5f;
        float scale = std::max({extents.x(), extents.y(), extents.z(), 1.0f}) / 32767.0f;
        
        // Collect unique vertices from triangles
        std::vector<Vector3> vertices;
        for (int idx : triIndices) {
            const CollisionTri_t& tri = g_collisionTris[idx];
            vertices.push_back(tri.v0);
            vertices.push_back(tri.v1);
            vertices.push_back(tri.v2);
        }
        
        int numVerts = std::min((int)vertices.size(), 255);
        int numFaces = std::min((int)triIndices.size(), 255);
        
        // Header: numVerts, numFaces, numTris, numPoly5
        uint32_t header = numVerts | (numFaces << 8) | (numFaces << 16) | (0 << 24);
        ApexLegends::Bsp::bvhLeafDatas.push_back(static_cast<int32_t>(header));
        
        // Origin and scale (as float bits stored in int32)
        union { float f; int32_t i; } conv;
        conv.f = center.x(); ApexLegends::Bsp::bvhLeafDatas.push_back(conv.i);
        conv.f = center.y(); ApexLegends::Bsp::bvhLeafDatas.push_back(conv.i);
        conv.f = center.z(); ApexLegends::Bsp::bvhLeafDatas.push_back(conv.i);
        conv.f = scale;      ApexLegends::Bsp::bvhLeafDatas.push_back(conv.i);
        
        // Packed vertices (simplified - each vert as 2 int32s containing 3 int16s)
        for (int i = 0; i < numVerts && i < (int)vertices.size(); i++) {
            Vector3 localPos = (vertices[i] - center) / scale;
            int16_t px = static_cast<int16_t>(std::clamp(localPos.x(), -32768.0f, 32767.0f));
            int16_t py = static_cast<int16_t>(std::clamp(localPos.y(), -32768.0f, 32767.0f));
            int16_t pz = static_cast<int16_t>(std::clamp(localPos.z(), -32768.0f, 32767.0f));
            
            // Pack 3 int16s into 2 int32s (48 bits total, 6 bytes)
            ApexLegends::Bsp::bvhLeafDatas.push_back((px) | (py << 16));
            if ((i & 1) == 0) {
                // Second half of packing
                ApexLegends::Bsp::bvhLeafDatas.push_back(pz);
            }
        }
        
        return leafIndex;
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
        
        Note: We always use packed vertices (bvhFlags=1), so we must use type 5 (POLY3)
        or higher. Type 4 (TRISTRIP) uses float vertices which we don't emit.
    */
    int SelectBestLeafType(const std::vector<int>& triIndices) {
        int numTris = static_cast<int>(triIndices.size());
        
        if (numTris == 0) {
            return BVH4_TYPE_EMPTY;
        }
        
        // Always use POLY3 (type 5) for packed vertices
        // TRISTRIP (type 4) uses float vertices which we don't support
        if (numTris <= 16) {
            return BVH4_TYPE_POLY3;
        }
        
        // For larger groups, use ConvexHull (most general)
        return BVH4_TYPE_CONVEXHULL;
    }

    /*
        EmitLeafDataForType
        Emits leaf data based on the specified type
        Returns index into bvhLeafDatas
    */
    int EmitLeafDataForType(int leafType, const std::vector<int>& triIndices, int surfPropIdx = 0) {
        switch (leafType) {
            case BVH4_TYPE_EMPTY:
                return 0;
                
            case BVH4_TYPE_TRISTRIP:
                return EmitTriangleStripLeaf(triIndices, surfPropIdx);
                
            case BVH4_TYPE_POLY3:
                return EmitPoly3Leaf(triIndices, surfPropIdx);
                
            case BVH4_TYPE_POLY4:
                return EmitPoly4Leaf(triIndices, surfPropIdx);
                
            case BVH4_TYPE_POLY5PLUS:
                // Fall through to ConvexHull for complex polys
            case BVH4_TYPE_CONVEXHULL:
            default:
                return EmitConvexHullLeaf(triIndices, surfPropIdx);
        }
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
        DistanceToSegmentSquared
        Returns squared distance from point to line segment
        Used for T-junction detection
    */
    float DistanceToSegmentSquared(const Vector3& point, const Vector3& segStart, const Vector3& segEnd) {
        Vector3 segDir = segEnd - segStart;
        float segLen = vector3_length(segDir);

        if (segLen < 0.0001f) {
            // Degenerate segment, return distance to endpoint
            return vector3_length_squared(point - segStart);
        }

        Vector3 segDirNorm = segDir / segLen;
        Vector3 toPoint = point - segStart;
        float projection = vector3_dot(toPoint, segDirNorm);

        // Clamp to segment endpoints
        float clampedProjection = std::clamp(projection, 0.0f, segLen);

        // Closest point on segment
        Vector3 closestPoint = segStart + segDirNorm * clampedProjection;

        return vector3_length_squared(point - closestPoint);
    }

    /*
        WeldVertexWithTJunctions
        Snaps a vertex to nearby existing vertices OR to existing edges
        This properly eliminates T-junctions that cause players to get stuck
        Now with grid snapping to eliminate floating point precision issues
    */
    Vector3 WeldVertexWithTJunctions(const Vector3& vert, std::vector<Vector3>& weldedVerts,
                                      std::vector<std::pair<Vector3, Vector3>>& weldedEdges) {
        // First snap to grid to eliminate precision issues
        Vector3 snappedVert = SnapVertexToGrid(vert);

        // First check if close to existing vertex
        for (const Vector3& existing : weldedVerts) {
            float dist = vector3_length(snappedVert - existing);
            if (dist < VERTEX_WELD_EPSILON) {
                return existing;  // Snap to existing vertex
            }
        }

        // Check if close to any existing edge (T-junction elimination)
        float epsilonSquared = VERTEX_WELD_EPSILON * VERTEX_WELD_EPSILON;
        for (const auto& edge : weldedEdges) {
            float distSq = DistanceToSegmentSquared(snappedVert, edge.first, edge.second);
            if (distSq < epsilonSquared) {
                // Snap this vertex to the edge
                // Project onto the line segment
                Vector3 segDir = edge.second - edge.first;
                float segLen = vector3_length(segDir);

                if (segLen < 0.0001f) {
                    // Degenerate edge, snap to start
                    Vector3 snapped = SnapVertexToGrid(edge.first);
                    weldedVerts.push_back(snapped);
                    return snapped;
                }

                Vector3 toVert = snappedVert - edge.first;
                Vector3 segDirNorm = segDir / segLen;
                float projection = vector3_dot(toVert, segDirNorm);

                // Clamp to segment and compute closest point
                float clampedProjection = std::clamp(projection, 0.0f, segLen);
                Vector3 closestPoint = edge.first + segDirNorm * clampedProjection;

                // Snap the closest point to grid as well
                closestPoint = SnapVertexToGrid(closestPoint);

                // Add this new split point to vertices
                weldedVerts.push_back(closestPoint);
                return closestPoint;
            }
        }

        // Not close to anything, add as new vertex (already snapped)
        weldedVerts.push_back(snappedVert);
        return snappedVert;
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
            
            Sys_Printf("  Node %d: Internal node with children [%d, %d, %d, %d]\n", 
                       bspNodeIndex, buildNode.childIndices[0], buildNode.childIndices[1],
                       buildNode.childIndices[2], buildNode.childIndices[3]);
            
            for (int i = 0; i < 4; i++) {
                if (buildNode.childIndices[i] >= 0) {
                    const BVHBuildNode_t& childBuild = g_bvhBuildNodes[buildNode.childIndices[i]];
                    childBounds[i] = childBuild.bounds;
                    Sys_Printf("    Child %d: bounds (%.1f,%.1f,%.1f)-(%.1f,%.1f,%.1f)\n", i,
                               childBounds[i].mins.x(), childBounds[i].mins.y(), childBounds[i].mins.z(),
                               childBounds[i].maxs.x(), childBounds[i].maxs.y(), childBounds[i].maxs.z());
                } else {
                    // Empty slot - use parent bounds
                    childBounds[i] = buildNode.bounds;
                    Sys_Printf("    Child %d: empty (using parent bounds)\n", i);
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
                
                Sys_Printf("    Setting child %d: type=%d, index=%d\n", i, childType, childIndex);
                
                // Access via index since vector may have reallocated during recursion
                switch (i) {
                    case 0: ApexLegends::Bsp::bvhNodes[bspNodeIndex].index0 = childIndex; break;
                    case 1: ApexLegends::Bsp::bvhNodes[bspNodeIndex].index1 = childIndex; break;
                    case 2: ApexLegends::Bsp::bvhNodes[bspNodeIndex].index2 = childIndex; break;
                    case 3: ApexLegends::Bsp::bvhNodes[bspNodeIndex].index3 = childIndex; break;
                }
            }
            
            // Debug: print the final node state
            Sys_Printf("    Final node %d: types=[%d,%d,%d,%d] indices=[%d,%d,%d,%d]\n",
                       bspNodeIndex,
                       ApexLegends::Bsp::bvhNodes[bspNodeIndex].childType0,
                       ApexLegends::Bsp::bvhNodes[bspNodeIndex].childType1,
                       ApexLegends::Bsp::bvhNodes[bspNodeIndex].childType2,
                       ApexLegends::Bsp::bvhNodes[bspNodeIndex].childType3,
                       ApexLegends::Bsp::bvhNodes[bspNodeIndex].index0,
                       ApexLegends::Bsp::bvhNodes[bspNodeIndex].index1,
                       ApexLegends::Bsp::bvhNodes[bspNodeIndex].index2,
                       ApexLegends::Bsp::bvhNodes[bspNodeIndex].index3);
            
            // Hex dump of the metadata dwords for verification
            const uint32_t* nodeData = reinterpret_cast<const uint32_t*>(&ApexLegends::Bsp::bvhNodes[bspNodeIndex]);
            Sys_Printf("    Metadata dwords: [0x%08X, 0x%08X, 0x%08X, 0x%08X]\n",
                       nodeData[12], nodeData[13], nodeData[14], nodeData[15]);
        }

        return bspNodeIndex;
    }

    /*
        CollectTrianglesFromMeshes
        Collects collision triangles from Shared::meshes
        Filters out degenerate triangles and welds close vertices
        Now includes T-junction elimination to prevent players getting stuck on edges
    */
    void CollectTrianglesFromMeshes() {
        g_collisionTris.clear();

        // Track welded vertices AND edges for proper T-junction elimination
        std::vector<Vector3> weldedVertices;
        std::vector<std::pair<Vector3, Vector3>> weldedEdges;
        int skippedDegenerate = 0;
        int totalTris = 0;
        int tJunctionsFixed = 0;

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
                // Weld vertices with T-junction elimination
                // This snaps vertices to nearby edges, preventing cracks
                Vector3 origV0 = verts[indices[i]].xyz;
                Vector3 origV1 = verts[indices[i + 1]].xyz;
                Vector3 origV2 = verts[indices[i + 2]].xyz;

                tri.v0 = WeldVertexWithTJunctions(origV0, weldedVertices, weldedEdges);
                tri.v1 = WeldVertexWithTJunctions(origV1, weldedVertices, weldedEdges);
                tri.v2 = WeldVertexWithTJunctions(origV2, weldedVertices, weldedEdges);

                // Track if any vertex was snapped to an edge (T-junction fixed)
                if (tri.v0 != origV0 || tri.v1 != origV1 || tri.v2 != origV2) {
                    // Check if it was snapped to an edge (not just vertex welding)
                    float distToV0 = vector3_length(tri.v0 - origV0);
                    float distToV1 = vector3_length(tri.v1 - origV1);
                    float distToV2 = vector3_length(tri.v2 - origV2);
                    float maxDist = std::max({distToV0, distToV1, distToV2});

                    if (maxDist > VERTEX_WELD_EPSILON * 0.5f) {
                        tJunctionsFixed++;
                    }
                }

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

                // Add this triangle's edges to the welded edges list for T-junction detection
                // Use consistent ordering (min vertex first) to avoid duplicates
                auto addEdge = [&](const Vector3& a, const Vector3& b) {
                    if (vector3_length(a - b) > VERTEX_WELD_EPSILON) {
                        // Sort vertices to ensure consistent edge direction
                        if (a.x() < b.x() || (a.x() == b.x() && a.y() < b.y()) ||
                            (a.x() == b.x() && a.y() == b.y() && a.z() < b.z())) {
                            weldedEdges.push_back({a, b});
                        } else {
                            weldedEdges.push_back({b, a});
                        }
                    }
                };

                addEdge(tri.v0, tri.v1);
                addEdge(tri.v1, tri.v2);
                addEdge(tri.v2, tri.v0);
            }
        }

        if (skippedDegenerate > 0) {
            Sys_FPrintf(SYS_VRB, "  Filtered %d degenerate triangles (%.1f%%)\n",
                       skippedDegenerate, 100.0f * skippedDegenerate / totalTris);
        }
        if (tJunctionsFixed > 0) {
            Sys_FPrintf(SYS_VRB, "  Fixed %d T-junctions (cracks in collision)\n", tJunctionsFixed);
        }
        Sys_Printf("  Collected %zu collision triangles (from %d total, %zu edges tracked)\n",
                   g_collisionTris.size(), totalTris, weldedEdges.size());
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
    
    Sys_Printf("  BVH bounds: mins(%.1f, %.1f, %.1f) maxs(%.1f, %.1f, %.1f)\n",
               overallBounds.mins.x(), overallBounds.mins.y(), overallBounds.mins.z(),
               overallBounds.maxs.x(), overallBounds.maxs.y(), overallBounds.maxs.z());
    Sys_Printf("  BVH origin: (%.1f, %.1f, %.1f), maxExtent: %.1f, scale: %g (1:1=%s)\n", 
               center.x(), center.y(), center.z(), maxExtent, bvhScale,
               (bvhScale == 1.0f/65536.0f) ? "yes" : "no");
    
    // Build triangle index list
    std::vector<int> allTriIndices(g_collisionTris.size());
    std::iota(allTriIndices.begin(), allTriIndices.end(), 0);
    
    // Build BVH tree
    g_bvhBuildNodes.clear();
    int rootBuildIndex = BuildBVH4Node(allTriIndices, 0);
    
    // Debug: dump all build nodes
    Sys_Printf("  Built %zu build nodes:\n", g_bvhBuildNodes.size());
    for (size_t n = 0; n < g_bvhBuildNodes.size(); n++) {
        const BVHBuildNode_t& node = g_bvhBuildNodes[n];
        Sys_Printf("    Node %zu: isLeaf=%d, bounds=(%.1f,%.1f,%.1f)-(%.1f,%.1f,%.1f), tris=%zu\n",
                   n, node.isLeaf,
                   node.bounds.mins.x(), node.bounds.mins.y(), node.bounds.mins.z(),
                   node.bounds.maxs.x(), node.bounds.maxs.y(), node.bounds.maxs.z(),
                   node.triangleIndices.size());
        if (!node.isLeaf) {
            Sys_Printf("            children: [%d, %d, %d, %d]\n",
                       node.childIndices[0], node.childIndices[1], 
                       node.childIndices[2], node.childIndices[3]);
        }
    }
    
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
    
    Sys_Printf("  Built BVH with %zu intermediate nodes\n", g_bvhBuildNodes.size());
    
    // Emit to BSP format
    int leafDataOffset = 0;
    int rootNodeIndex = EmitBVH4Nodes(rootBuildIndex, leafDataOffset);
    
    Sys_Printf("  Emitted %zu BVH nodes, %zu leaf data entries, %zu packed vertices\n", 
               ApexLegends::Bsp::bvhNodes.size(),
               ApexLegends::Bsp::bvhLeafDatas.size(),
               ApexLegends::Bsp::packedVertices.size());
    Sys_Printf("  Model vertexIndex: %d, bvhFlags: %d\n", 
               model.vertexIndex, model.bvhFlags);
    
    // Verification: dump first few packed vertices
    Sys_Printf("  First packed vertices (hex):\n");
    for (size_t i = 0; i < std::min((size_t)6, ApexLegends::Bsp::packedVertices.size()); i++) {
        const auto& v = ApexLegends::Bsp::packedVertices[i];
        Sys_Printf("    V%zu: x=%04X y=%04X z=%04X\n", i, 
                   (uint16_t)v.x, (uint16_t)v.y, (uint16_t)v.z);
    }
    
    // Verification: dump first few leaf data entries
    Sys_Printf("  First leaf data entries (hex):\n");
    for (size_t i = 0; i < std::min((size_t)8, ApexLegends::Bsp::bvhLeafDatas.size()); i++) {
        Sys_Printf("    [%zu]: 0x%08X\n", i, (uint32_t)ApexLegends::Bsp::bvhLeafDatas[i]);
    }
    
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