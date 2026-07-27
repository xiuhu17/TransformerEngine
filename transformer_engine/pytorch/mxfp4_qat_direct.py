# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Direct MXFP4 -> FP8 weight conversion for MXFP4 QAT (TileKernels canonicalization).

Converts the MXFP4 decomposition of a bf16/fp32 master weight straight into
the host recipe's FP8 representation without a bf16 bridge tensor:

* MXFP8 rowwise (tex.mxfp4_direct_mxfp8_rowwise): the deployment fixed-shift-6
  canonicalization -- scale code = fp4_exp + 121 clamped at UE8M0 code 0 with
  exact payload absorption; the bytes are invertible back to packed FP4.
* MXFP8 columnwise (backward_override=None only): the literal
  projection-then-requantize chain on existing kernels --
  tex.mxfp4_fake_quantize produces the dequantized MXFP4-grid weight and the
  host MXFP8 columnwise quantizer encodes it, so the columnwise buffers are
  the bridge path's, byte for byte, by construction.
* Float8 blockwise 128x128 (tex.mxfp4_direct_blockwise): per-tile exponent
  folding, RTNE onto E4M3 with subnormals -- exact through scale spread 2^14,
  bounded beyond, no device assert.

Decoded values equal the bridge path (mxfp4_fake_quantize) everywhere both
are defined; only the rowwise/blockwise raw bytes differ (fixed-shift vs
amax canonicalization). The numerical specification lives in
tests/pytorch/references/mxfp4_qat_direct_reference.py.
"""
import torch

import transformer_engine_torch as tex

from .mxfp4_qat import _MXFP4_BLOCK, mxfp4_fake_quantize
from .tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor
from .tensor.float8_blockwise_tensor import Float8BlockQuantizer, Float8BlockwiseQTensor

__all__ = ["mxfp4_qat_direct_quantize", "mxfp4_qat_direct_update_"]


def _require_kernel(name: str):
    fn = getattr(tex, name, None)
    if fn is None:
        raise RuntimeError(
            f"transformer_engine_torch was built without {name}; rebuild Transformer Engine."
        )
    return fn


def _prepare(weight: torch.Tensor) -> torch.Tensor:
    w = weight.contiguous()
    if w.data_ptr() % 16 != 0:
        w = w.clone()
    return w


def _mxfp8_columnwise_from_projection(weight: torch.Tensor):
    """Columnwise (backward_override=None) buffers via the spec-literal chain
    bf16 -> mxfp4(row) -> dequantized bf16 -> host MXFP8 columnwise quantize.

    Both steps run existing kernels (the QAT projection and the production
    columnwise encoder), so the result is bit-identical to the bridge path's
    columnwise representation by construction.
    """
    what = mxfp4_fake_quantize(weight)
    col = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=False, columnwise=True).quantize(
        what
    )
    return col._columnwise_data, col._columnwise_scale_inv


def mxfp4_qat_direct_quantize(weight: torch.Tensor, quantizer) -> torch.Tensor:
    """Convert a bf16/fp32 weight straight to the host FP8 representation."""
    if not weight.is_cuda:
        raise ValueError("MXFP4 QAT direct conversion expects a CUDA weight tensor")
    if weight.dim() != 2:
        raise ValueError(f"expected a 2D weight, got {tuple(weight.shape)}")
    if weight.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError(f"bf16/fp32 weights only, got {weight.dtype}")
    rows, cols = weight.shape
    if cols % _MXFP4_BLOCK != 0:
        raise ValueError(f"inner dim must be divisible by {_MXFP4_BLOCK}, got {cols}")

    if isinstance(quantizer, MXFP8Quantizer):
        data, scale_inv = _require_kernel("mxfp4_direct_mxfp8_rowwise")(_prepare(weight))
        col_data, col_scale_inv = None, None
        if quantizer.columnwise_usage:
            if rows % 32 != 0:
                raise ValueError(f"columnwise 32x1 blocks need rows % 32 == 0, got {rows}")
            col_data, col_scale_inv = _mxfp8_columnwise_from_projection(weight)
        return MXFP8Tensor(
            shape=weight.shape,
            dtype=weight.dtype,
            rowwise_data=data if quantizer.rowwise_usage else None,
            rowwise_scale_inv=scale_inv if quantizer.rowwise_usage else None,
            columnwise_data=col_data,
            columnwise_scale_inv=col_scale_inv,
            fp8_dtype=tex.DType.kFloat8E4M3,
            quantizer=quantizer,
            with_gemm_swizzled_scales=False,
            device=weight.device,
        )

    if isinstance(quantizer, Float8BlockQuantizer):
        if getattr(quantizer, "block_scaling_dim", 2) != 2:
            raise ValueError("direct blockwise conversion supports 128x128 (2D) scaling only")
        data, scale_inv = _require_kernel("mxfp4_direct_blockwise")(_prepare(weight))
        out = Float8BlockwiseQTensor(
            shape=weight.shape,
            dtype=weight.dtype,
            fp8_dtype=tex.DType.kFloat8E4M3,
            rowwise_data=data,
            rowwise_scale_inv=scale_inv,
            columnwise_data=None,
            columnwise_scale_inv=None,
            quantizer=quantizer,
            is_2D_scaled=True,
            device=weight.device,
        )
        if quantizer.columnwise_usage:
            out._create_columnwise()  # exact transpose of the rowwise encoding
        return out

    raise NotImplementedError(
        f"MXFP4 QAT direct conversion does not support {type(quantizer).__name__}"
    )


def mxfp4_qat_direct_update_(weight: torch.Tensor, workspace) -> None:
    """Requantize ``weight`` into an existing workspace's buffers in place."""
    quantizer = workspace._quantizer
    fresh = mxfp4_qat_direct_quantize(weight, quantizer)
    if isinstance(workspace, MXFP8Tensor):
        if workspace._rowwise_data is not None:
            workspace._rowwise_data.copy_(fresh._rowwise_data)
            workspace._rowwise_scale_inv.copy_(fresh._rowwise_scale_inv)
        if workspace._columnwise_data is not None:
            workspace._columnwise_data.copy_(fresh._columnwise_data)
            workspace._columnwise_scale_inv.copy_(fresh._columnwise_scale_inv)
        return
    if isinstance(workspace, Float8BlockwiseQTensor):
        if workspace._rowwise_data is not None:
            workspace._rowwise_data.copy_(fresh._rowwise_data)
            workspace._rowwise_scale_inv.copy_(fresh._rowwise_scale_inv)
        if workspace._columnwise_data is not None:
            fresh._create_columnwise()
            workspace._columnwise_data.copy_(fresh._columnwise_data)
            workspace._columnwise_scale_inv.copy_(fresh._columnwise_scale_inv)
        return
    raise NotImplementedError(
        f"MXFP4 QAT direct update does not support {type(workspace).__name__}"
    )
