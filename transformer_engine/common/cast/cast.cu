/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/cast.h>
#include <transformer_engine/multi_stream.h>

#include "../common.h"
#include "../transpose/cast_transpose.h"
#include "../util/multi_stream.h"
#include "../utils.cuh"
#include "dispatch/dequantize.cuh"
#include "dispatch/quantize.cuh"
#include "mxfp4/direct_convert_mxfp4.cuh"
#include "mxfp4/fake_quantize_mxfp4.cuh"
#include "transformer_engine/transpose.h"

void nvte_quantize(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize);
  using namespace transformer_engine;

  constexpr bool IS_ACT = false;
  dispatch::quantize_fwd_helper<IS_ACT, Empty, nullptr>(input, output, nullptr, stream);
}

void nvte_quantize_noop(const NVTETensor input, NVTETensor output, NVTETensor noop,
                        cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_noop);
  using namespace transformer_engine;

  // Create config with noop tensor
  QuantizationConfig quant_config;
  quant_config.noop_tensor = noop;

  nvte_quantize_v2(input, output, reinterpret_cast<NVTEQuantizationConfig>(&quant_config), stream);
}

void nvte_quantize_v2(const NVTETensor input, NVTETensor output,
                      const NVTEQuantizationConfig quant_config, cudaStream_t stream) {
  NVTE_API_CALL(nvte_quantize_v2);
  using namespace transformer_engine;

  constexpr bool IS_ACT = false;
  dispatch::quantize_fwd_helper<IS_ACT, Empty, nullptr>(input, output, quant_config, stream);
}

void nvte_mxfp4_fake_quantize(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_mxfp4_fake_quantize);
  using namespace transformer_engine;
  const Tensor &input_tensor = *convertNVTETensorCheck(input);
  Tensor &output_tensor = *convertNVTETensorCheck(output);
  NVTE_CHECK(input_tensor.dtype() == DType::kBFloat16 || input_tensor.dtype() == DType::kFloat32,
             "MXFP4 fake-quantize supports BF16/FP32 inputs, got ",
             to_string(input_tensor.dtype()));
  NVTE_CHECK(output_tensor.dtype() == input_tensor.dtype(),
             "MXFP4 fake-quantize output dtype must match the input dtype.");
  NVTE_CHECK(input_tensor.data.shape == output_tensor.data.shape,
             "MXFP4 fake-quantize output shape must match the input shape.");
  const size_t numel = input_tensor.numel();
  NVTE_CHECK(numel % 32 == 0, "MXFP4 fake-quantize needs the element count divisible by 32.");
  NVTE_CHECK(!input_tensor.data.shape.empty() && input_tensor.data.shape.back() % 32 == 0,
             "MXFP4 fake-quantize needs the innermost dimension divisible by 32 "
             "(1x32 blocks must not cross rows).");
  if (numel == 0) {
    return;
  }
  if (input_tensor.dtype() == DType::kBFloat16) {
    dispatch::mxfp4::fake_quantize_mxfp4_launch<bf16>(
        reinterpret_cast<const bf16 *>(input_tensor.data.dptr),
        reinterpret_cast<bf16 *>(output_tensor.data.dptr), numel, stream);
  } else {
    dispatch::mxfp4::fake_quantize_mxfp4_launch<fp32>(
        reinterpret_cast<const fp32 *>(input_tensor.data.dptr),
        reinterpret_cast<fp32 *>(output_tensor.data.dptr), numel, stream);
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void nvte_mxfp4_direct_mxfp8_rowwise(const NVTETensor input, NVTETensor data, NVTETensor scales,
                                     cudaStream_t stream) {
  NVTE_API_CALL(nvte_mxfp4_direct_mxfp8_rowwise);
  using namespace transformer_engine;
  const auto &input_tensor = *convertNVTETensorCheck(input);
  auto &data_tensor = *convertNVTETensorCheck(data);
  auto &scales_tensor = *convertNVTETensorCheck(scales);
  if (input_tensor.data.numel() == 0) return;
  NVTE_CHECK(input_tensor.data.shape.size() == 2, "MXFP4 direct rowwise expects a 2D input.");
  const size_t rows = input_tensor.data.shape[0], cols = input_tensor.data.shape[1];
  NVTE_CHECK(cols % 32 == 0, "MXFP4 direct rowwise needs the innermost dim divisible by 32.");
  NVTE_CHECK(data_tensor.data.dtype == DType::kByte && scales_tensor.data.dtype == DType::kByte,
             "MXFP4 direct rowwise outputs must be uint8.");
  const int scale_stride = static_cast<int>(scales_tensor.data.shape[1]);
  switch (input_tensor.data.dtype) {
    case DType::kBFloat16:
      dispatch::mxfp4::direct_mxfp8_rowwise_launch<bf16>(
          reinterpret_cast<const bf16 *>(input_tensor.data.dptr),
          reinterpret_cast<uint8_t *>(data_tensor.data.dptr),
          reinterpret_cast<uint8_t *>(scales_tensor.data.dptr), rows, cols, scale_stride, stream);
      break;
    case DType::kFloat32:
      dispatch::mxfp4::direct_mxfp8_rowwise_launch<fp32>(
          reinterpret_cast<const fp32 *>(input_tensor.data.dptr),
          reinterpret_cast<uint8_t *>(data_tensor.data.dptr),
          reinterpret_cast<uint8_t *>(scales_tensor.data.dptr), rows, cols, scale_stride, stream);
      break;
    default:
      NVTE_ERROR("MXFP4 direct conversion supports BF16/FP32 inputs only.");
  }
}

void nvte_mxfp4_direct_blockwise(const NVTETensor input, NVTETensor data, NVTETensor scales,
                                 cudaStream_t stream) {
  NVTE_API_CALL(nvte_mxfp4_direct_blockwise);
  using namespace transformer_engine;
  const auto &input_tensor = *convertNVTETensorCheck(input);
  auto &data_tensor = *convertNVTETensorCheck(data);
  auto &scales_tensor = *convertNVTETensorCheck(scales);
  if (input_tensor.data.numel() == 0) return;
  NVTE_CHECK(input_tensor.data.shape.size() == 2, "MXFP4 direct blockwise expects a 2D input.");
  const size_t rows = input_tensor.data.shape[0], cols = input_tensor.data.shape[1];
  NVTE_CHECK(rows % 128 == 0 && cols % 128 == 0,
             "MXFP4 direct blockwise needs 128-divisible dimensions.");
  NVTE_CHECK(scales_tensor.data.dtype == DType::kFloat32,
             "MXFP4 direct blockwise scales must be fp32.");
  const int scale_stride = static_cast<int>(scales_tensor.data.shape[1]);
  switch (input_tensor.data.dtype) {
    case DType::kBFloat16:
      dispatch::mxfp4::direct_blockwise_launch<bf16>(
          reinterpret_cast<const bf16 *>(input_tensor.data.dptr),
          reinterpret_cast<uint8_t *>(data_tensor.data.dptr),
          reinterpret_cast<float *>(scales_tensor.data.dptr), rows, cols, scale_stride, stream);
      break;
    case DType::kFloat32:
      dispatch::mxfp4::direct_blockwise_launch<fp32>(
          reinterpret_cast<const fp32 *>(input_tensor.data.dptr),
          reinterpret_cast<uint8_t *>(data_tensor.data.dptr),
          reinterpret_cast<float *>(scales_tensor.data.dptr), rows, cols, scale_stride, stream);
      break;
    default:
      NVTE_ERROR("MXFP4 direct conversion supports BF16/FP32 inputs only.");
  }
}

void nvte_dequantize(const NVTETensor input, NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_dequantize);
  using namespace transformer_engine;
  dispatch::dequantize_helper(*convertNVTETensorCheck(input), convertNVTETensorCheck(output),
                              stream);
}

void nvte_multi_tensor_quantize(const NVTETensor *inputs, NVTETensor *outputs,
                                const NVTEQuantizationConfig quant_configs,
                                const size_t num_tensors, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_quantize);
  using namespace transformer_engine;

  constexpr bool IS_ACT = false;

  const size_t num_streams = nvte_get_num_compute_streams();

  int num_stream_used = std::min(num_streams, num_tensors);
  // wait for current stream to finish
  NVTE_CHECK_CUDA(cudaEventRecord(detail::get_compute_stream_event(0), stream));
  for (int s = 0; s < num_stream_used; s++) {
    NVTE_CHECK_CUDA(
        cudaStreamWaitEvent(detail::get_compute_stream(s), detail::get_compute_stream_event(0)));
  }

  for (int i = 0; i < num_tensors; i++) {
    dispatch::quantize_fwd_helper<IS_ACT, Empty, nullptr>(
        inputs[i], outputs[i], quant_configs, detail::get_compute_stream(i % num_streams));
  }

  // record events on compute streams
  for (int s = 0; s < num_stream_used; s++) {
    NVTE_CHECK_CUDA(
        cudaEventRecord(detail::get_compute_stream_event(s), detail::get_compute_stream(s)));
  }
  // wait for all compute streams to finish
  for (int s = 0; s < num_stream_used; s++) {
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream, detail::get_compute_stream_event(s)));
  }
}
