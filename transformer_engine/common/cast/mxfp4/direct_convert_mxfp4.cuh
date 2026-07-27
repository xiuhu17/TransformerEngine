/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file direct_convert_mxfp4.cuh
 *  \brief Direct MXFP4 -> FP8 weight conversion kernels (MXFP4 QAT).
 *
 *  Converts a bf16/fp32 master weight straight into the host recipe's FP8
 *  representation without materializing the dequantized MXFP4-grid weight:
 *
 *   - rowwise MXFP8: the TileKernels deployment canonicalization. Per 1x32
 *     block, scale code = fp4_exp + 121 (fixed shift 6) clamped at UE8M0
 *     code 0 with exact payload absorption; E4M3 payload = E2M1 payload
 *     * 2^shift, in [32, 384] (or [1, 192] in the clamped domain).
 *   - blockwise 128x128: the per_block_cast_lossless exponent folding. Tile
 *     scale = 2^(max block exponent - 6) clamped at 2^-127 (the fp32-stored
 *     scale stays on the E8M0 lattice Blackwell extracts); folded payloads
 *     round RTNE onto E4M3 (subnormals included) -- no 2^11 device assert:
 *     exact through spread 2^14, bounded rounding beyond.
 *
 *  Value-domain semantics (bit-identical to the torch reference in
 *  transformer_engine/pytorch/mxfp4_qat_direct.py): all-zero blocks pin the
 *  fp4 exponent at -126 (TileKernels clamp); non-finite 1x32 blocks emit
 *  E4M3 NaN payloads (0x7F) with scale code 127 (rowwise); the rowwise
 *  payload keeps the sign of -0.
 */

#ifndef TRANSFORMER_ENGINE_CAST_MXFP4_DIRECT_CONVERT_MXFP4_CUH_
#define TRANSFORMER_ENGINE_CAST_MXFP4_DIRECT_CONVERT_MXFP4_CUH_

#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "fake_quantize_mxfp4.cuh"

namespace transformer_engine {
namespace dispatch {
namespace mxfp4 {
namespace direct_convert_kernel {

using fake_quantize_kernel::block_scale_exponent_from_bits;
using fake_quantize_kernel::E2M1_MAX;
using fake_quantize_kernel::exp2_from_int;
using fake_quantize_kernel::mul_rn_no_ftz;
using fake_quantize_kernel::MXFP4_BLOCK_SIZE;
using fake_quantize_kernel::round_e2m1_magnitude;
using fake_quantize_kernel::THREADS_PER_CHUNK;

// 2^k as fp32 for k in [-149, 127]: normal via exponent field, subnormal via
// mantissa bit, exact zero below the subnormal range. Integer construction.
__device__ __forceinline__ float pow2_f32_ext(const int k) {
  if (k >= -126) {
    return exp2_from_int(k);
  }
  if (k >= -149) {
    return __int_as_float(1 << (k + 149));
  }
  return 0.0f;
}

// E4M3 byte of one exact payload value (sign of zero preserved, RTNE): the
// hardware cvt.rn.satfinite.e4m3x2 instruction via the CUDA fp8 intrinsics.
__device__ __forceinline__ void store_e4m3_pair(uint8_t *dst, const float lo, const float hi) {
  const __nv_fp8x2_storage_t two =
      __nv_cvt_float2_to_fp8x2(make_float2(lo, hi), __NV_SATFINITE, __NV_E4M3);
  dst[0] = static_cast<uint8_t>(two & 0xFF);
  dst[1] = static_cast<uint8_t>(two >> 8);
}

// One element: MXFP4 RTNE at scale 2^e, then payload value * 2^shift.
template <typename IType>
__device__ __forceinline__ float direct_payload(const IType v, const float inv_scale,
                                                const float pscale) {
  const float vf = static_cast<float>(v);
  const float av = __int_as_float(__float_as_int(vf) & 0x7fffffff);
  const float y = fminf(mul_rn_no_ftz(av, inv_scale), E2M1_MAX);
  const float q = copysignf(round_e2m1_magnitude(y), vf);
  return mul_rn_no_ftz(q, pscale);
}

// ---------------------------------------------------------------- rowwise
// Same shape of work as the fake-quantize kernel: each thread owns one
// 16-byte input vector; aligned lane groups reduce the block amax over the
// fp32 bit patterns; every lane emits its payload bytes and the group leader
// writes the scale code into the padded (roundup(M,128), roundup(N/32,4))
// layout.
template <typename IType>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    direct_mxfp8_rowwise_kernel(const IType *__restrict__ input, uint8_t *__restrict__ data,
                                uint8_t *__restrict__ scales, const size_t num_vectors,
                                const int blocks_per_row, const int scale_stride) {
  constexpr size_t VEC_ELTS = 16 / sizeof(IType);
  constexpr int LANES_PER_BLOCK = MXFP4_BLOCK_SIZE / VEC_ELTS;

  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t vec_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       vec_idx < num_vectors; vec_idx += stride) {
    Vec<IType, VEC_ELTS> in_vec;
    in_vec.load_from(input, vec_idx);

    int amax_bits = 0;
#pragma unroll
    for (int i = 0; i < static_cast<int>(VEC_ELTS); ++i) {
      amax_bits =
          max(amax_bits, __float_as_int(static_cast<float>(in_vec.data.elt[i])) & 0x7fffffff);
    }
    const unsigned lane = threadIdx.x & 31u;
    const unsigned group_mask = (LANES_PER_BLOCK == 32) ? 0xffffffffu
                                                        : (((1u << LANES_PER_BLOCK) - 1u)
                                                           << (lane & ~(LANES_PER_BLOCK - 1u)));
#pragma unroll
    for (int offset = LANES_PER_BLOCK / 2; offset > 0; offset >>= 1) {
      amax_bits = max(amax_bits, __shfl_xor_sync(group_mask, amax_bits, offset));
    }
    const bool nonfinite = amax_bits >= 0x7f800000;

    // fp4 exponent (zero/subnormal amax pins at -126 = the TileKernels clamp);
    // fixed-shift-6 lift with exact payload absorption below UE8M0 code 0
    const int e = block_scale_exponent_from_bits(amax_bits);
    const int biased = e + 121;
    const int shift = biased >= 0 ? 6 : e + 127;
    const float inv_scale = exp2_from_int(-e);
    const float pscale = exp2_from_int(shift);

    Vec<uint8_t, VEC_ELTS> out_vec;
    if (nonfinite) {
#pragma unroll
      for (int i = 0; i < static_cast<int>(VEC_ELTS); ++i) {
        out_vec.data.elt[i] = 0x7F;  // E4M3 NaN payload
      }
    } else {
#pragma unroll
      for (int i = 0; i < static_cast<int>(VEC_ELTS); i += 2) {
        store_e4m3_pair(&out_vec.data.elt[i], direct_payload(in_vec.data.elt[i], inv_scale, pscale),
                        direct_payload(in_vec.data.elt[i + 1], inv_scale, pscale));
      }
    }
    out_vec.store_to(data, vec_idx);

    if ((vec_idx % LANES_PER_BLOCK) == 0) {
      const size_t block_idx = vec_idx / LANES_PER_BLOCK;
      const size_t r = block_idx / blocks_per_row;
      const size_t c = block_idx % blocks_per_row;
      scales[r * scale_stride + c] =
          nonfinite ? static_cast<uint8_t>(127) : static_cast<uint8_t>(max(biased, 0));
    }
  }
}

// ---------------------------------------------------------------- blockwise
// One CTA per 128x128 tile, one thread per 1x32 block (512 threads). Stage 1
// derives every block's fp4 exponent; a shared-memory tree reduce yields the
// tile maximum; stage 2 folds each block's payloads by 2^(e - tile_exp) and
// encodes RTNE onto E4M3 (subnormal payload codes included).
template <typename IType>
__global__ void __launch_bounds__(512)
    direct_blockwise_kernel(const IType *__restrict__ input, uint8_t *__restrict__ data,
                            float *__restrict__ scales, const size_t cols, const int scale_stride) {
  constexpr int BLOCKS_PER_TILE_ROW = 128 / MXFP4_BLOCK_SIZE;  // 4

  const int t = threadIdx.x;
  const size_t row = static_cast<size_t>(blockIdx.y) * 128 + (t / BLOCKS_PER_TILE_ROW);
  const size_t col0 = static_cast<size_t>(blockIdx.x) * 128 +
                      static_cast<size_t>(t % BLOCKS_PER_TILE_ROW) * MXFP4_BLOCK_SIZE;
  const IType *src = input + row * cols + col0;

  float elts[MXFP4_BLOCK_SIZE];
  int amax_bits = 0;
#pragma unroll
  for (int i = 0; i < static_cast<int>(MXFP4_BLOCK_SIZE); ++i) {
    elts[i] = static_cast<float>(src[i]);
    amax_bits = max(amax_bits, __float_as_int(elts[i]) & 0x7fffffff);
  }
  const bool nonfinite = amax_bits >= 0x7f800000;
  const int e = block_scale_exponent_from_bits(amax_bits);

  __shared__ int e_sh[512];
  e_sh[t] = nonfinite ? -126 : e;  // non-finite blocks do not raise the tile max
  __syncthreads();
#pragma unroll
  for (int s = 256; s > 0; s >>= 1) {
    if (t < s) {
      e_sh[t] = max(e_sh[t], e_sh[t + s]);
    }
    __syncthreads();
  }
  const int s_exp = max(e_sh[0] - 6, -127);

  uint8_t *dst = data + row * cols + col0;
  if (nonfinite) {
#pragma unroll
    for (int i = 0; i < static_cast<int>(MXFP4_BLOCK_SIZE); ++i) {
      dst[i] = 0x7F;
    }
  } else {
    const float inv_scale = exp2_from_int(-e);
    const float pscale = pow2_f32_ext(e - s_exp);
#pragma unroll
    for (int i = 0; i < static_cast<int>(MXFP4_BLOCK_SIZE); i += 2) {
      uint8_t pair[2];
      const float av0 = __int_as_float(__float_as_int(elts[i]) & 0x7fffffff);
      const float av1 = __int_as_float(__float_as_int(elts[i + 1]) & 0x7fffffff);
      const float q0 =
          copysignf(round_e2m1_magnitude(fminf(mul_rn_no_ftz(av0, inv_scale), E2M1_MAX)), elts[i]);
      const float q1 = copysignf(
          round_e2m1_magnitude(fminf(mul_rn_no_ftz(av1, inv_scale), E2M1_MAX)), elts[i + 1]);
      store_e4m3_pair(pair, mul_rn_no_ftz(q0, pscale), mul_rn_no_ftz(q1, pscale));
      dst[i] = pair[0];
      dst[i + 1] = pair[1];
    }
  }

  if (t == 0) {
    scales[static_cast<size_t>(blockIdx.y) * scale_stride + blockIdx.x] =
        (s_exp == -127) ? __int_as_float(0x00400000) : exp2_from_int(s_exp);
  }
}

}  // namespace direct_convert_kernel

// Raw-pointer launchers. Callers guarantee contiguous tensors, 16-byte
// aligned input/data pointers, cols % 32 == 0 (rowwise) or rows/cols % 128
// (blockwise), and zero-initialized padded scale buffers.
template <typename IType>
void direct_mxfp8_rowwise_launch(const IType *input, uint8_t *data, uint8_t *scales,
                                 const size_t rows, const size_t cols, const int scale_stride,
                                 cudaStream_t stream) {
  using namespace direct_convert_kernel;
  constexpr size_t VEC_ELTS = 16 / sizeof(IType);
  const size_t num_vectors = rows * cols / VEC_ELTS;
  const size_t blocks_needed = (num_vectors + THREADS_PER_CHUNK - 1) / THREADS_PER_CHUNK;
  const int grid = static_cast<int>(blocks_needed < 65535 ? blocks_needed : 65535);
  direct_mxfp8_rowwise_kernel<IType><<<grid, THREADS_PER_CHUNK, 0, stream>>>(
      input, data, scales, num_vectors, static_cast<int>(cols / MXFP4_BLOCK_SIZE), scale_stride);
}

template <typename IType>
void direct_blockwise_launch(const IType *input, uint8_t *data, float *scales, const size_t rows,
                             const size_t cols, const int scale_stride, cudaStream_t stream) {
  using namespace direct_convert_kernel;
  const dim3 grid(static_cast<unsigned>(cols / 128), static_cast<unsigned>(rows / 128));
  direct_blockwise_kernel<IType><<<grid, 512, 0, stream>>>(input, data, scales, cols, scale_stride);
}

}  // namespace mxfp4
}  // namespace dispatch
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_CAST_MXFP4_DIRECT_CONVERT_MXFP4_CUH_
