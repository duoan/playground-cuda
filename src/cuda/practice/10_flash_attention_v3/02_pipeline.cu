// practice: flash_attention_v3 version = pipeline
// TODO: Same warp-per-row + two-slot K/V shared-memory double buffer (preload / consume).
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../common/cuda_utils.cuh"

namespace {

constexpr int kQueryCount = 4;
constexpr int kKeyCount = 16;
constexpr int kHeadDim = 8;
constexpr float kEps = 1e-6f;

double dot_product(const float* a, const float* b, int dim) {
    double result = 0.0;
    for (int d = 0; d < dim; ++d) result += static_cast<double>(a[d]) * static_cast<double>(b[d]);
    return result;
}

void fill_inputs(std::vector<float>& q, std::vector<float>& k, std::vector<float>& v) {
    for (int row = 0; row < kQueryCount; ++row)
        for (int d = 0; d < kHeadDim; ++d)
            q[row * kHeadDim + d] = 0.1f * static_cast<float>(row + 1) + 0.03f * static_cast<float>(d);
    for (int row = 0; row < kKeyCount; ++row)
        for (int d = 0; d < kHeadDim; ++d) {
            k[row * kHeadDim + d] = 0.05f * static_cast<float>(row + 1) + 0.02f * static_cast<float>(d + 1);
            v[row * kHeadDim + d] = 0.07f * static_cast<float>(row + 1) + 0.01f * static_cast<float>(d);
        }
}

void flash_attention_cpu(const std::vector<float>& q, const std::vector<float>& k,
                         const std::vector<float>& v, std::vector<float>& out) {
    for (int row = 0; row < kQueryCount; ++row) {
        double scores[kKeyCount];
        double row_max = -1.0e30;
        for (int key = 0; key < kKeyCount; ++key) {
            const double s = dot_product(&q[row * kHeadDim], &k[key * kHeadDim], kHeadDim);
            scores[key] = s;
            if (s > row_max) row_max = s;
        }
        double row_sum = 0.0;
        double accum[kHeadDim] = {0.0};
        for (int key = 0; key < kKeyCount; ++key) {
            const double w = std::exp(scores[key] - row_max);
            row_sum += w;
            for (int d = 0; d < kHeadDim; ++d) accum[d] += w * static_cast<double>(v[key * kHeadDim + d]);
        }
        for (int d = 0; d < kHeadDim; ++d) out[row * kHeadDim + d] = static_cast<float>(accum[d] / (row_sum + kEps));
    }
}

bool check(const std::vector<float>& got, const std::vector<float>& expected) {
    for (size_t i = 0; i < got.size(); ++i)
        if (std::fabs(got[i] - expected[i]) > 1e-4f) {
            std::cerr << "mismatch " << i << ": " << got[i] << " vs " << expected[i] << '\n';
            return false;
        }
    return true;
}

// TODO: kernel(s) for this version.

// TODO: launch. See ../../10_flash_attention_v3/02_pipeline.cu for the reference.
void launch(const float* dq, const float* dk, const float* dv, float* dout) {
    (void)dq; (void)dk; (void)dv; (void)dout;
}

}  // namespace

int main() {
    std::vector<float> q(kQueryCount * kHeadDim), k(kKeyCount * kHeadDim), v(kKeyCount * kHeadDim);
    std::vector<float> out(kQueryCount * kHeadDim, 0.0f), ref(kQueryCount * kHeadDim, 0.0f);
    fill_inputs(q, k, v);
    flash_attention_cpu(q, k, v, ref);

    float *dq, *dk, *dv, *dout;
    const size_t qb = q.size() * sizeof(float), kb = k.size() * sizeof(float);
    const size_t vb = v.size() * sizeof(float), ob = out.size() * sizeof(float);
    CHECK_CUDA(cudaMalloc(&dq, qb));
    CHECK_CUDA(cudaMalloc(&dk, kb));
    CHECK_CUDA(cudaMalloc(&dv, vb));
    CHECK_CUDA(cudaMalloc(&dout, ob));
    CHECK_CUDA(cudaMemcpy(dq, q.data(), qb, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dk, k.data(), kb, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dv, v.data(), vb, cudaMemcpyHostToDevice));

    launch(dq, dk, dv, dout);

    CHECK_CUDA(cudaMemcpy(out.data(), dout, ob, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(dq));
    CHECK_CUDA(cudaFree(dk));
    CHECK_CUDA(cudaFree(dv));
    CHECK_CUDA(cudaFree(dout));

    if (!check(out, ref)) return EXIT_FAILURE;
    std::cout << "flash_attention_v3 [pipeline] practice PASS  q=" << kQueryCount
              << " k=" << kKeyCount << " d=" << kHeadDim << '\n';
    return EXIT_SUCCESS;
}
