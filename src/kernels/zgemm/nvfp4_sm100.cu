#include "zgemm.h"
#include "../utils.cuh"
#include "../dispatch_utils.h"

// Kernels around the cuBLAS NVFP4 GEMM that runs SVDQuant layers on data-center Blackwell (sm_100a / sm_103a).

namespace nunchaku::kernels {

namespace {

// One thread per group of 16 values, with the arithmetic of quantize_w4a4_fp4_from_fpsum_warp in gemm_w4a4.cuh:
// divide by the smoothing factor and round to 16 bits, scale = min(absmax / 6, 448) stored as e4m3, multiply by the
// approximate reciprocal of the unrounded scale and convert to E2M1. The output is in the layout cuBLAS reads: codes
// row-major with the first value in the low nibble, scales in 128 x 4 tiles stored as 32 x 16.
template<typename T>
__global__ void
quantize_nvfp4_act_kernel(const T *input, const T *smooth, uint8_t *output, uint8_t *oscales, int M, int K) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
    constexpr int GROUP = 16;
    const int numGroups = K / GROUP;
    const int64_t i     = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)M * numGroups) {
        return;
    }
    const int row   = i / numGroups;
    const int group = i % numGroups;

    const uint4 *src = reinterpret_cast<const uint4 *>(input + (int64_t)row * K + group * GROUP);
    const uint4 *sf  = reinterpret_cast<const uint4 *>(smooth + group * GROUP);
    uint4 rx[2]      = {src[0], src[1]};
    uint4 rs[2]      = {__ldg(&sf[0]), __ldg(&sf[1])};
    const T *x       = reinterpret_cast<const T *>(rx);
    const T *s       = reinterpret_cast<const T *>(rs);

    float v[GROUP];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < GROUP; j++) {
        v[j] = cuda_cast<float>(cuda_cast<T>(__fdividef(cuda_cast<float>(x[j]), cuda_cast<float>(s[j]))));
        amax = fmaxf(amax, fabsf(v[j]));
    }

    const float scale = fminf(amax * (1.0f / 6.0f), 448.0f);
    float rscale      = 0.0f;
    if (scale > 0.0f) {
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(rscale) : "f"(scale));
    }
    uint16_t scale_e4m3x2;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(scale_e4m3x2) : "f"(0.0f), "f"(scale));

    uint32_t codes[2] = {0, 0};
#pragma unroll
    for (int j = 0; j < GROUP; j += 2) {
        uint32_t byte;
        asm("{ .reg .b8 tmp; cvt.rn.satfinite.e2m1x2.f32 tmp, %1, %2; cvt.u32.u8 %0, tmp; }"
            : "=r"(byte)
            : "f"(v[j + 1] * rscale), "f"(v[j] * rscale));
        codes[j / 8] |= byte << (j % 8 * 4);
    }
    *reinterpret_cast<uint2 *>(output + (int64_t)row * (K / 2) + group * (GROUP / 2)) = make_uint2(codes[0], codes[1]);

    const int r = row % 128, c = group % 4;
    const int64_t tile                                 = (int64_t)(row / 128) * (K / 64) + group / 4;
    oscales[tile * 512 + r % 32 * 16 + r / 32 * 4 + c] = scale_e4m3x2 & 0xff;
#else
    trap_unsupported_arch();
#endif
}

// RMSNorm over each 128-wide head, then rotary embedding on adjacent pairs, in place on the Q and K parts of a
// [M, 3 * H * 128] QKV tensor; same arithmetic as EpilogueRMSNormRope. One warp per head, 4 values per lane.
template<typename T>
__global__ void qk_rmsnorm_rope_kernel(T *qkv,
                                       const T *normQ,
                                       const T *normK,
                                       const float *rotaryEmb, // pack_rotemb layout
                                       int M,
                                       int numHeads,
                                       int seqLen,
                                       int rotaryBatch,
                                       int rotaryTokens,
                                       float eps) {
    constexpr int HEAD_DIM = 128;
    const int warpId       = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int laneId       = threadIdx.x % 32;
    if (warpId >= M * numHeads * 2) {
        return;
    }
    const int row  = warpId / (numHeads * 2);
    const int isK  = warpId % (numHeads * 2) / numHeads;
    const int head = warpId % numHeads;

    T *ptr        = qkv + ((int64_t)row * 3 + isK) * numHeads * HEAD_DIM + head * HEAD_DIM + laneId * 4;
    uint2 rx      = *reinterpret_cast<uint2 *>(ptr);
    const T *data = reinterpret_cast<const T *>(&rx);
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        v[j] = cuda_cast<float>(data[j]);
    }

    const T *norm = isK ? normK : normQ;
    if (norm != nullptr) {
        float sum = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
#pragma unroll
        for (int mask = 16; mask > 0; mask /= 2) {
            sum += __shfl_xor_sync(~0u, sum, mask);
        }
        const float rms = rsqrtf(sum / HEAD_DIM + eps);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            v[j] *= rms * cuda_cast<float>(norm[laneId * 4 + j]);
        }
    }

    if (rotaryEmb != nullptr) {
        // (B, M / 16, D / 8, 8 [m % 8], 4 [pair % 4], 2 [m % 16 / 8], 2 [sin, cos])
        const int m      = row % seqLen;
        const float *emb = rotaryEmb + (int64_t)(rotaryBatch == 1 ? 0 : row / seqLen) * rotaryTokens * HEAD_DIM;
#pragma unroll
        for (int p = 0; p < 2; p++) {
            const int pair    = laneId * 2 + p;
            const int64_t idx = (((int64_t)(m / 16) * (HEAD_DIM / 8) + pair / 4) * 8 + m % 8) * 4 + pair % 4;
            const float sin   = emb[(idx * 2 + m % 16 / 8) * 2];
            const float cos   = emb[(idx * 2 + m % 16 / 8) * 2 + 1];
            const float x = v[p * 2], y = v[p * 2 + 1];
            v[p * 2]     = x * cos - y * sin;
            v[p * 2 + 1] = x * sin + y * cos;
        }
    }

    T out[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
        out[j] = cuda_cast<T>(v[j]);
    }
    *reinterpret_cast<uint2 *>(ptr) = *reinterpret_cast<uint2 *>(out);
}

// out = out * wcscales + bias per column, in place, rounded once as in EpilogueBias; 8 values per thread
template<typename T>
__global__ void scale_bias_kernel(T *out, const T *wcscales, const T *bias, int64_t numel, int N) {
    const int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= numel) {
        return;
    }
    const int col = i % N;
    uint4 rx      = *reinterpret_cast<uint4 *>(out + i);
    uint4 rs      = __ldg(reinterpret_cast<const uint4 *>(wcscales + col));
    uint4 rb      = bias != nullptr ? __ldg(reinterpret_cast<const uint4 *>(bias + col)) : make_uint4(0, 0, 0, 0);
    const T *x = reinterpret_cast<const T *>(&rx), *s = reinterpret_cast<const T *>(&rs),
            *b = reinterpret_cast<const T *>(&rb);
    T y[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        y[j] = cuda_cast<T>(fmaf(cuda_cast<float>(x[j]), cuda_cast<float>(s[j]), cuda_cast<float>(b[j])));
    }
    *reinterpret_cast<uint4 *>(out + i) = *reinterpret_cast<uint4 *>(y);
}

} // namespace

void quantize_nvfp4_act_sm100(Tensor input, Tensor smooth, Tensor output, Tensor oscales) {
    const int K = input.shape[-1];
    const int M = input.numel() / K;
    assert(K % 64 == 0);
    assert(smooth.numel() == K && smooth.dtype() == input.dtype());
    assert(output.dtype() == Tensor::INT8 && output.numel() == (size_t)M * K / 2);
    assert(oscales.dtype() == Tensor::FP8_E4M3 && oscales.numel() == (size_t)ceilDiv(M, 128) * 128 * (K / 16));

    auto stream = getCurrentCUDAStream();
    if (M % 128 != 0) {
        // cuBLAS reads the scales of the padding rows in the last 128-row tile
        const size_t tileBytes = (size_t)(K / 64) * 512;
        checkCUDA(cudaMemsetAsync(oscales.data_ptr<uint8_t>() + (M / 128) * tileBytes, 0, tileBytes, stream));
    }

    const int threadsPerBlock = 256;
    const int blocksPerGrid   = ceilDiv<int64_t>((int64_t)M * (K / 16), threadsPerBlock);
    dispatchFloat16(input.scalar_type(), [&]<typename scalar_t>() {
        quantize_nvfp4_act_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(input.data_ptr<scalar_t>(),
                                                                                 smooth.data_ptr<scalar_t>(),
                                                                                 output.data_ptr<uint8_t>(),
                                                                                 oscales.data_ptr<uint8_t>(),
                                                                                 M,
                                                                                 K);
    });
    checkCUDA(cudaGetLastError());
}

void qk_rmsnorm_rope_sm100(Tensor qkv, Tensor norm_q, Tensor norm_k, Tensor rotary_emb, int seq_len, float eps) {
    constexpr int HEAD_DIM = 128;
    const int N            = qkv.shape[-1];
    const int M            = qkv.numel() / N;
    assert(N % (3 * HEAD_DIM) == 0);
    assert(M % seq_len == 0);
    int rotaryBatch = 1, rotaryTokens = 0;
    if (rotary_emb.valid()) {
        assert(rotary_emb.dtype() == Tensor::FP32 && rotary_emb.ndims() == 3 && rotary_emb.shape[2] == HEAD_DIM);
        rotaryBatch  = rotary_emb.shape[0];
        rotaryTokens = rotary_emb.shape[1];
        assert(rotaryTokens >= seq_len && rotaryTokens % 16 == 0);
    }
    const int numHeads = N / (3 * HEAD_DIM);

    const int threadsPerBlock = 256;
    const int blocksPerGrid   = ceilDiv<int64_t>((int64_t)M * numHeads * 2 * 32, threadsPerBlock);
    auto stream               = getCurrentCUDAStream();
    dispatchFloat16(qkv.scalar_type(), [&]<typename scalar_t>() {
        qk_rmsnorm_rope_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            qkv.data_ptr<scalar_t>(),
            norm_q.valid() ? norm_q.data_ptr<scalar_t>() : nullptr,
            norm_k.valid() ? norm_k.data_ptr<scalar_t>() : nullptr,
            rotary_emb.valid() ? rotary_emb.data_ptr<float>() : nullptr,
            M,
            numHeads,
            seq_len,
            rotaryBatch,
            rotaryTokens,
            eps);
    });
    checkCUDA(cudaGetLastError());
}

void scale_bias_sm100(Tensor out, Tensor wcscales, Tensor bias) {
    const int N = out.shape[-1];
    assert(N % 8 == 0);
    assert(wcscales.numel() == N && wcscales.dtype() == out.dtype());
    assert(!bias.valid() || (bias.numel() == N && bias.dtype() == out.dtype()));

    const int threadsPerBlock = 256;
    const int blocksPerGrid   = ceilDiv<int64_t>(out.numel() / 8, threadsPerBlock);
    auto stream               = getCurrentCUDAStream();
    dispatchFloat16(out.scalar_type(), [&]<typename scalar_t>() {
        scale_bias_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(out.data_ptr<scalar_t>(),
                                                                         wcscales.data_ptr<scalar_t>(),
                                                                         bias.valid() ? bias.data_ptr<scalar_t>()
                                                                                      : nullptr,
                                                                         out.numel(),
                                                                         N);
    });
    checkCUDA(cudaGetLastError());
}

}; // namespace nunchaku::kernels
