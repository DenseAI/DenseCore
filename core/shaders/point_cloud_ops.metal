/**
 * @file point_cloud_ops.metal
 * @brief Metal kernels for point-cloud and NeRF-style universal ops
 *
 * These kernels prioritize correctness and UMA zero-copy interop on Apple
 * Silicon. They are scalar/reference-style GPU implementations, not final
 * hand-tuned kernels.
 */

#include <metal_stdlib>
using namespace metal;

#define MAX_K 64u

inline float distance_squared3(float ax, float ay, float az, float bx, float by, float bz) {
    const float dx = ax - bx;
    const float dy = ay - by;
    const float dz = az - bz;
    return dx * dx + dy * dy + dz * dz;
}

inline uint expand_bits_10(uint v) {
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v << 8)) & 0x0300F00Fu;
    v = (v | (v << 4)) & 0x030C30C3u;
    v = (v | (v << 2)) & 0x09249249u;
    return v;
}

inline uint morton_code3d(float x, float y, float z, float voxel_size) {
    const uint ix = uint(min(max(x / voxel_size, 0.0f), 1023.0f));
    const uint iy = uint(min(max(y / voxel_size, 0.0f), 1023.0f));
    const uint iz = uint(min(max(z / voxel_size, 0.0f), 1023.0f));
    return (expand_bits_10(iz) << 2) | (expand_bits_10(iy) << 1) | expand_bits_10(ix);
}

inline uint morton_rank_for_point(device const float* points, uint batch, uint num_points, uint point_idx,
                                  float voxel_size) {
    const uint point_base = ((batch * num_points) + point_idx) * 3u;
    const uint point_code =
        morton_code3d(points[point_base + 0], points[point_base + 1], points[point_base + 2], voxel_size);

    uint rank = 0u;
    for (uint other = 0u; other < num_points; ++other) {
        const uint other_base = ((batch * num_points) + other) * 3u;
        const uint other_code =
            morton_code3d(points[other_base + 0], points[other_base + 1], points[other_base + 2], voxel_size);
        if (other_code < point_code || (other_code == point_code && other < point_idx)) {
            ++rank;
        }
    }
    return rank;
}

kernel void point_cloud_patchify_forward(
    device const float* points [[buffer(0)]],
    device const float* features [[buffer(1)]],
    device float* patches [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& num_points [[buffer(4)]],
    constant uint& feature_dim [[buffer(5)]],
    constant uint& num_patches [[buffer(6)]],
    constant uint& patch_dim [[buffer(7)]],
    constant float& voxel_size [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint dim = gid.x;
    const uint patch = gid.y;
    const uint b = gid.z;
    if (b >= batch || patch >= num_patches || dim >= patch_dim) return;

    const uint points_per_patch = max(1u, num_points / max(1u, num_patches));
    const uint start = patch * points_per_patch;
    const uint end = min(start + points_per_patch, num_points);
    const uint count = end > start ? end - start : 0u;

    float sum = 0.0f;
    if (dim < feature_dim) {
        for (uint i = 0u; i < num_points; ++i) {
            const uint rank = morton_rank_for_point(points, b, num_points, i, voxel_size);
            if (rank < start || rank >= end) continue;
            const uint idx = ((b * num_points + i) * feature_dim) + dim;
            sum += features[idx];
        }
    }

    patches[((b * num_patches + patch) * patch_dim) + dim] =
        (dim < feature_dim && count > 0u) ? (sum / float(count)) : 0.0f;
}

kernel void point_cloud_unpatchify_forward(
    device const float* patches [[buffer(0)]],
    device const float* points [[buffer(1)]],
    device float* features [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& num_points [[buffer(4)]],
    constant uint& feature_dim [[buffer(5)]],
    constant uint& num_patches [[buffer(6)]],
    constant float& voxel_size [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint dim = gid.x;
    const uint point = gid.y;
    const uint b = gid.z;
    if (b >= batch || point >= num_points || dim >= feature_dim) return;

    const uint points_per_patch = max(1u, num_points / max(1u, num_patches));
    const uint assigned_points = min(num_points, points_per_patch * num_patches);
    const uint rank = morton_rank_for_point(points, b, num_points, point, voxel_size);
    if (rank >= assigned_points) return;

    const uint patch = min(rank / points_per_patch, num_patches - 1u);
    features[((b * num_points + point) * feature_dim) + dim] =
        patches[((b * num_patches + patch) * feature_dim) + dim];
}

kernel void nerf_positional_encoding_forward(
    device const float* xyz [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& num_points [[buffer(3)]],
    constant uint& freq_bands [[buffer(4)]],
    constant uint& include_input [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint out_dim = gid.x;
    const uint point = gid.y;
    const uint b = gid.z;
    const uint input_dims = include_input ? 3u : 0u;
    const uint total_dim = input_dims + 6u * freq_bands;
    if (b >= batch || point >= num_points || out_dim >= total_dim) return;

    const uint src_base = (b * num_points + point) * 3u;
    if (include_input && out_dim < 3u) {
        output[(b * num_points + point) * total_dim + out_dim] = xyz[src_base + out_dim];
        return;
    }

    const uint encoded_idx = out_dim - input_dims;
    const uint band = encoded_idx / 6u;
    const uint axis = (encoded_idx % 6u) / 2u;
    const bool use_cos = (encoded_idx % 2u) == 1u;
    const float freq = exp2(float(band));
    const float angle = xyz[src_base + axis] * freq;
    output[(b * num_points + point) * total_dim + out_dim] = use_cos ? cos(angle) : sin(angle);
}

kernel void gaussian_fourier_features_forward(
    device const float* xyz [[buffer(0)]],
    device const float* B_matrix [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& num_points [[buffer(4)]],
    constant uint& num_features [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint feature = gid.x;
    const uint point = gid.y;
    const uint b = gid.z;
    if (b >= batch || point >= num_points || feature >= num_features) return;

    const uint half_features = num_features / 2u;
    const uint freq_idx = feature % half_features;
    const bool use_cos = feature >= half_features;
    const uint xyz_base = (b * num_points + point) * 3u;
    const uint mat_base = freq_idx * 3u;
    const float projection =
        xyz[xyz_base + 0] * B_matrix[mat_base + 0] +
        xyz[xyz_base + 1] * B_matrix[mat_base + 1] +
        xyz[xyz_base + 2] * B_matrix[mat_base + 2];
    output[(b * num_points + point) * num_features + feature] = use_cos ? cos(projection) : sin(projection);
}

kernel void farthest_point_sampling_forward(
    device const float* points [[buffer(0)]],
    device int* indices [[buffer(1)]],
    constant uint& batch [[buffer(2)]],
    constant uint& num_points [[buffer(3)]],
    constant uint& num_samples [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= batch) return;
    const uint b = gid;
    const uint sample_count = min(num_samples, num_points);
    const uint idx_base = b * num_samples;
    indices[idx_base] = 0;

    for (uint s = 1; s < sample_count; ++s) {
        float best_dist = -1.0f;
        uint best_idx = 0;
        for (uint i = 0; i < num_points; ++i) {
            const uint point_base = (b * num_points + i) * 3u;
            const float px = points[point_base + 0];
            const float py = points[point_base + 1];
            const float pz = points[point_base + 2];

            float min_dist = INFINITY;
            for (uint prev = 0; prev < s; ++prev) {
                const uint selected = uint(indices[idx_base + prev]);
                const uint sel_base = (b * num_points + selected) * 3u;
                min_dist = min(min_dist, distance_squared3(
                    px, py, pz,
                    points[sel_base + 0], points[sel_base + 1], points[sel_base + 2]));
            }
            if (min_dist > best_dist) {
                best_dist = min_dist;
                best_idx = i;
            }
        }
        indices[idx_base + s] = int(best_idx);
    }

    for (uint s = sample_count; s < num_samples; ++s) {
        indices[idx_base + s] = indices[idx_base + (s % sample_count)];
    }
}

kernel void knn_query_forward(
    device const float* query_points [[buffer(0)]],
    device const float* reference_points [[buffer(1)]],
    device int* indices [[buffer(2)]],
    device float* distances [[buffer(3)]],
    constant uint& batch [[buffer(4)]],
    constant uint& num_queries [[buffer(5)]],
    constant uint& num_references [[buffer(6)]],
    constant uint& k [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint q = gid.x;
    const uint b = gid.z;
    if (b >= batch || q >= num_queries || k == 0u || k > MAX_K) return;

    thread float best_dist[MAX_K];
    thread int best_idx[MAX_K];
    for (uint i = 0; i < k; ++i) {
        best_dist[i] = INFINITY;
        best_idx[i] = -1;
    }

    const uint query_base = (b * num_queries + q) * 3u;
    const float qx = query_points[query_base + 0];
    const float qy = query_points[query_base + 1];
    const float qz = query_points[query_base + 2];

    for (uint r = 0; r < num_references; ++r) {
        const uint ref_base = (b * num_references + r) * 3u;
        const float dist = distance_squared3(
            qx, qy, qz,
            reference_points[ref_base + 0],
            reference_points[ref_base + 1],
            reference_points[ref_base + 2]);

        uint insert_at = k;
        for (uint i = 0; i < k; ++i) {
            if (dist < best_dist[i] || (dist == best_dist[i] && int(r) < best_idx[i])) {
                insert_at = i;
                break;
            }
        }
        if (insert_at == k) continue;

        for (uint i = k - 1u; i > insert_at; --i) {
            best_dist[i] = best_dist[i - 1u];
            best_idx[i] = best_idx[i - 1u];
        }
        best_dist[insert_at] = dist;
        best_idx[insert_at] = int(r);
    }

    const uint out_base = (b * num_queries + q) * k;
    for (uint i = 0; i < k; ++i) {
        const uint fill = (best_idx[i] >= 0) ? i : max(0u, i > 0u ? i - 1u : 0u);
        indices[out_base + i] = best_idx[i] >= 0 ? best_idx[i] : best_idx[fill];
        distances[out_base + i] = isfinite(best_dist[i]) ? best_dist[i] : best_dist[fill];
    }
}

kernel void ball_query_forward(
    device const float* query_points [[buffer(0)]],
    device const float* reference_points [[buffer(1)]],
    device int* indices [[buffer(2)]],
    constant uint& batch [[buffer(3)]],
    constant uint& num_queries [[buffer(4)]],
    constant uint& num_references [[buffer(5)]],
    constant float& radius [[buffer(6)]],
    constant uint& max_samples [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint q = gid.x;
    const uint b = gid.z;
    if (b >= batch || q >= num_queries || max_samples == 0u) return;

    const float radius2 = radius * radius;
    const uint out_base = (b * num_queries + q) * max_samples;
    const uint query_base = (b * num_queries + q) * 3u;
    const float qx = query_points[query_base + 0];
    const float qy = query_points[query_base + 1];
    const float qz = query_points[query_base + 2];

    uint count = 0u;
    for (uint r = 0; r < num_references && count < max_samples; ++r) {
        const uint ref_base = (b * num_references + r) * 3u;
        if (distance_squared3(qx, qy, qz,
                              reference_points[ref_base + 0],
                              reference_points[ref_base + 1],
                              reference_points[ref_base + 2]) <= radius2) {
            indices[out_base + count] = int(r);
            ++count;
        }
    }

    for (uint i = count; i < max_samples; ++i) {
        indices[out_base + i] = -1;
    }
}

kernel void deformable_attention_3d_forward(
    device const float* query [[buffer(0)]],
    device const float* key [[buffer(1)]],
    device const float* value [[buffer(2)]],
    device const float* reference_points [[buffer(3)]],
    device const float* sampling_offsets [[buffer(4)]],
    device const float* attention_weights [[buffer(5)]],
    device const float* key_points [[buffer(6)]],
    device float* output [[buffer(7)]],
    constant uint& batch [[buffer(8)]],
    constant uint& num_queries [[buffer(9)]],
    constant uint& num_points_key [[buffer(10)]],
    constant uint& hidden_dim [[buffer(11)]],
    constant uint& num_heads [[buffer(12)]],
    constant uint& total_samples [[buffer(13)]],
    constant uint& offset_heads [[buffer(14)]],
    constant uint& weight_heads [[buffer(15)]],
    constant uint& use_key_points [[buffer(16)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint d = gid.x;
    const uint q = gid.y;
    const uint b = gid.z;
    if (b >= batch || q >= num_queries || d >= hidden_dim || num_heads == 0u) return;
    (void)query;

    const uint d_per_head = max(1u, hidden_dim / num_heads);
    const uint head = min(d / d_per_head, num_heads - 1u);
    const uint offset_head = offset_heads > 1u ? head : 0u;
    const uint weight_head = weight_heads > 1u ? head : 0u;

    const uint ref_base = (b * num_queries + q) * 3u;
    const float rx = reference_points[ref_base + 0];
    const float ry = reference_points[ref_base + 1];
    const float rz = reference_points[ref_base + 2];

    float sum_w = 0.0f;
    for (uint s = 0; s < total_samples; ++s) {
        const uint w_idx = (((b * num_queries + q) * weight_heads + weight_head) * total_samples) + s;
        sum_w += max(0.0f, attention_weights[w_idx]);
    }
    const float inv_sum_w = (sum_w > 1e-6f) ? (1.0f / sum_w) : (1.0f / float(max(1u, total_samples)));

    float out = 0.0f;
    for (uint s = 0; s < total_samples; ++s) {
        const uint off_idx = ((((b * num_queries + q) * offset_heads + offset_head) * total_samples) + s) * 3u;
        const float sx = rx + sampling_offsets[off_idx + 0];
        const float sy = ry + sampling_offsets[off_idx + 1];
        const float sz = rz + sampling_offsets[off_idx + 2];

        uint best_idx = 0u;
        float best_dist = INFINITY;
        for (uint n = 0; n < num_points_key; ++n) {
            const uint coord_base = use_key_points
                ? ((b * num_points_key + n) * 3u)
                : ((b * num_points_key + n) * hidden_dim);
            const float kx = use_key_points ? key_points[coord_base + 0] : key[coord_base + 0];
            const float ky = use_key_points ? key_points[coord_base + 1] : key[coord_base + 1];
            const float kz = use_key_points ? key_points[coord_base + 2] : key[coord_base + 2];
            const float dist = distance_squared3(sx, sy, sz, kx, ky, kz);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = n;
            }
        }

        const uint w_idx = (((b * num_queries + q) * weight_heads + weight_head) * total_samples) + s;
        const float w = max(0.0f, attention_weights[w_idx]) * inv_sum_w;
        out = fma(w, value[(b * num_points_key + best_idx) * hidden_dim + d], out);
    }

    output[(b * num_queries + q) * hidden_dim + d] = out;
}
