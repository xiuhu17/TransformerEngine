# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Direct MXFP4 -> FP8 weight conversion for MXFP4 QAT (TileKernels canonicalization).

Converts the MXFP4 decomposition of a bf16/fp32 master weight straight into
the host recipe's FP8 representation without a bf16 bridge tensor:

* MXFP8 rowwise (tex.mxfp4_direct_mxfp8_rowwise): the deployment fixed-shift-6
  canonicalization -- scale code = fp4_exp + 121 clamped at UE8M0 code 0 with
  exact payload absorption; the bytes are invertible back to packed FP4.
* MXFP8 columnwise (backward_override=None only): a fresh lossy 32x1
  quantization of the grid values with the HOST encoder's amax rule
  (scale = 2^ceil(log2(amax/448))), so dgrad bytes match the bridge path bit
  for bit. This is the sole implementation of the columnwise direction (a
  composite-torch computation, not a fallback for a kernel).
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

from .mxfp4_qat import _MXFP4_BLOCK
from .tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor
from .tensor.float8_blockwise_tensor import Float8BlockQuantizer, Float8BlockwiseQTensor

__all__ = ["mxfp4_qat_direct_quantize", "mxfp4_qat_direct_update_"]

_E4M3_NAN = 0x7F


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


def _roundup(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _round_to_e2m1_grid(y: torch.Tensor) -> torch.Tensor:
    """RTNE onto the E2M1 magnitude grid {0, .5, 1, 1.5, 2, 3, 4, 6}; input in [0, 6]."""
    fine = torch.round(y * 2.0) * 0.5
    mid = torch.round(y)
    coarse = torch.round(y * 0.5) * 2.0
    return torch.where(y <= 2.0, fine, torch.where(y <= 4.0, mid, coarse))


def _direct_mxfp8_columnwise(weight: torch.Tensor):
    """Sole (composite-torch) implementation of the columnwise direction.

    MXFP4-decomposes the weight, then quantizes the grid values in 32x1
    blocks with the host encoder's amax rule so the bytes match the bridge
    path's columnwise representation bit for bit.
    """
    rows, cols = weight.shape
    w32 = weight.contiguous().to(torch.float32).view(rows, cols // _MXFP4_BLOCK, _MXFP4_BLOCK)
    amax = w32.abs().amax(dim=-1, keepdim=True)
    nonfinite = ~torch.isfinite(amax)

    bits = amax.view(torch.int32)
    exp_field = bits >> 23
    mantissa = bits & 0x7FFFFF
    e = exp_field - 129 + (mantissa > 0x400000).to(torch.int32)
    e = torch.where(exp_field > 0, e, torch.full_like(e, -126))
    e = e.clamp(min=-126, max=125)
    e = torch.where(amax > 0, e, torch.full_like(e, -126))
    e = torch.where(nonfinite, torch.full_like(e, -126), e)

    scale = torch.ldexp(torch.ones_like(amax), e)
    y = (w32 / scale).abs().clamp(max=6.0)
    q = _round_to_e2m1_grid(torch.where(nonfinite, torch.zeros_like(y), y))
    q = torch.copysign(q, w32)

    vals = (q * scale).view(rows, cols)
    nf_elem = nonfinite.expand(-1, -1, _MXFP4_BLOCK).reshape(rows, cols)

    vc = vals.view(rows // 32, 32, cols)
    amax_c = vc.abs().amax(dim=1, keepdim=True)
    amax_c = torch.where(torch.isfinite(amax_c), amax_c, torch.zeros_like(amax_c))
    cb = amax_c.view(torch.int32)
    kf = (cb >> 23) - 127
    ec = kf - 8 + ((cb & 0x7FFFFF) > 0x600000).to(torch.int32)  # host rule: ceil(log2(amax/448))
    ec = torch.where((cb >> 23) > 0, ec, torch.full_like(ec, -127))
    ec = ec.clamp(min=-127, max=127)
    ec = torch.where(amax_c > 0, ec, torch.zeros_like(ec))
    inv = torch.ldexp(torch.ones_like(ec, dtype=torch.float32), -ec)
    # + 0.0 canonicalizes -0 to +0 (the host colwise kernel drops the sign of zero)
    payload = (vc * inv + 0.0).to(torch.float8_e4m3fn).view(torch.uint8).view(rows, cols)
    payload = torch.where(nf_elem, torch.full_like(payload, _E4M3_NAN), payload)

    code = (ec + 127).clamp(0, 254).to(torch.uint8)
    scale_inv = torch.zeros(
        (_roundup(rows // 32, 4), _roundup(cols, 128)), dtype=torch.uint8, device=payload.device
    )
    scale_inv[: rows // 32, :cols] = code.view(rows // 32, cols)
    return payload, scale_inv


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
            col_data, col_scale_inv = _direct_mxfp8_columnwise(weight)
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
