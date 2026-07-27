# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Direct MXFP4 -> FP8 weight conversion for MXFP4 QAT (TileKernels canonicalization).

Converts the MXFP4 decomposition of a bf16/fp32 master weight into the host
recipe's FP8 representation:

* MXFP8 rowwise (tex.mxfp4_direct_mxfp8_rowwise): one kernel reads the
  high-precision weight and writes the deployment fixed-shift-6
  canonicalization: for an MXFP4 block exponent e the scale is 2^(e-6), so the
  UE8M0 code is e + 121. The code is unsigned, so it saturates at 0 (2^-127) once
  e <= -121 and the payload absorbs the leftover power of two exactly. MXFP4 is
  never materialized. Above the saturation point the bytes invert back to packed
  FP4 by the fixed shift; at code 0 several (exponent, payload) pairs collide on
  one byte pair -- e.g. e=-126 with payload 6 and e=-125 with payload 3 both give
  payload 12 -- so the value survives but the original MXFP4 scale does not.
* MXFP8 columnwise (backward_override=None only): the spec's own chain, run on
  existing kernels -- tex.mxfp4_fake_quantize produces the dequantized
  MXFP4-grid weight and the host MXFP8 columnwise encoder quantizes it. This
  leg does materialize the projected weight, because the specified chain does.
  The 32x1 blocks run down columns, so the columnwise bytes are a second
  encoding of the same grid values rather than a transpose of the rowwise ones.
* Float8 blockwise 128x128 (tex.mxfp4_direct_blockwise): per-tile exponent
  folding at tile scale 2^(max block exponent - 6), RTNE onto E4M3 with
  subnormals -- exact through scale spread 2^14 when the tile's leading block
  saturates the E2M1 grid, bounded beyond, no device assert. Both dims must be
  128-divisible; the stock encoder pads partial tiles, this kernel does not.

Equivalence to the bridge path (mxfp4_fake_quantize followed by the host
quantizer), stated precisely:

* MXFP8 rowwise: decoded values are equal; the raw bytes differ by design
  (fixed-shift versus the host's amax canonicalization).
* MXFP8 columnwise: the bridge path's bytes exactly -- same encoder, same
  input, and the same GEMM scale swizzle when the host quantizer asks for one.
* Blockwise 128x128: decoded values agree when the tile's leading block reaches
  E2M1 payload 4 or 6, which is what the fixed shift of 6 assumes. When the
  leading payload is smaller the host picks a finer tile scale, so a tile that
  mixes a small leading payload with a large scale spread can keep a value the
  fixed-shift encoding rounds to zero. That is a real value difference, not a
  different factorization of the same value.

The numerical specification lives in
tests/pytorch/references/mxfp4_qat_direct_reference.py.
"""
import torch

import transformer_engine_torch as tex

from .mxfp4_qat import _MXFP4_BLOCK, mxfp4_fake_quantize
from .tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor
from .tensor.float8_blockwise_tensor import Float8BlockQuantizer, Float8BlockwiseQTensor

__all__ = ["mxfp4_qat_direct_quantize", "mxfp4_qat_direct_update_"]

_BLOCKWISE_TILE = 128


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
    columnwise encoder), and the encoder is asked for columnwise only, so the
    discarded rowwise half of the bridge path is never computed.

    On payload parity with the bridge: MXFP8 has a specialized fused kernel for
    the rowwise and bidimensional cases and a generic kernel for everything else,
    and only the specialized one canonicalizes -0 payloads to +0. The specialized
    kernel is skipped whenever the quantizer wants GEMM-swizzled scales, which is
    exactly what the modules request for MXFP8 weights (the one exception,
    primary FP8 weights, is rejected by MXFP4 QAT outright). So the host and this
    columnwise-only call both land on the generic kernel and agree byte for byte.

    Scales come back compact; the caller swizzles the whole tensor once when the
    host quantizer wants preswizzled scales, which the bridge encoder does inline.
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
        data, scale_inv = None, None
        if quantizer.rowwise_usage:
            data, scale_inv = _require_kernel("mxfp4_direct_mxfp8_rowwise")(_prepare(weight))
        col_data, col_scale_inv = None, None
        if quantizer.columnwise_usage:
            if rows % _MXFP4_BLOCK != 0:
                raise ValueError(
                    f"columnwise 32x1 blocks need rows % {_MXFP4_BLOCK} == 0, got {rows}"
                )
            # The rowwise bytes above cannot seed this: MXFP8 columnwise blocks run
            # 32x1 down columns, a different partition of the same matrix than the
            # 1x32 rowwise blocks, so the column amaxes are not recoverable from a
            # rowwise encoding. The projected weight has to exist in high precision
            # again, which is why the projection kernel runs a second time here.
            # The forward operand still comes from the direct kernel above so it
            # keeps the deployment canonicalization; a fused kernel emitting both
            # encodings in one pass would collapse these three launches into one.
            col_data, col_scale_inv = _mxfp8_columnwise_from_projection(weight)
        out = MXFP8Tensor(
            shape=weight.shape,
            dtype=weight.dtype,
            rowwise_data=data,
            rowwise_scale_inv=scale_inv,
            columnwise_data=col_data,
            columnwise_scale_inv=col_scale_inv,
            fp8_dtype=tex.DType.kFloat8E4M3,
            quantizer=quantizer,
            with_gemm_swizzled_scales=False,
            device=weight.device,
        )
        _apply_gemm_swizzle(out, quantizer)
        return out

    if isinstance(quantizer, Float8BlockQuantizer):
        if getattr(quantizer, "block_scaling_dim", 2) != 2:
            raise ValueError("direct blockwise conversion supports 128x128 (2D) scaling only")
        if rows % _BLOCKWISE_TILE != 0 or cols % _BLOCKWISE_TILE != 0:
            # The bridge path pads partial tiles; the direct kernel writes whole
            # 128x128 tiles only, so reject rather than emit a truncated scale grid.
            raise ValueError(
                "direct blockwise conversion needs both dims divisible by "
                f"{_BLOCKWISE_TILE}, got {tuple(weight.shape)}"
            )
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
        _apply_gemm_swizzle(out, quantizer)
        return out

    raise NotImplementedError(
        f"MXFP4 QAT direct conversion does not support {type(quantizer).__name__}"
    )


def _apply_gemm_swizzle(out, quantizer) -> None:
    """Bake the GEMM scale swizzle in when the host module asked for preswizzle.

    The module sets ``optimize_for_gemm`` on the weight quantizer, and the stock
    encoder honours it inside the cast kernel. The direct kernels always write
    compact scales, so swizzle here instead; without this the workspace layout
    would differ from the bridge path's and every GEMM would swizzle lazily.
    No-op for scaling modes that need no swizzle (e.g. FP8 block scaling).
    """
    if getattr(quantizer, "optimize_for_gemm", False):
        tex.swizzle_scales_for_gemm_(out)


def mxfp4_qat_direct_update_(weight: torch.Tensor, workspace, noop_flag=None) -> None:
    """Requantize ``weight`` into an existing workspace's buffers in place."""
    if noop_flag is not None:
        # The stock path forwards this device-side flag into the cast kernel so a
        # captured graph can skip the weight update; the direct kernels take no
        # such flag, and honouring it host-side would break graph capture.
        raise NotImplementedError(
            "MXFP4 QAT direct conversion does not support skipping the weight update "
            "(CUDA graph capture with cached FP8 weights). Use the projection mode."
        )
    if not isinstance(workspace, (MXFP8Tensor, Float8BlockwiseQTensor)):
        raise NotImplementedError(
            f"MXFP4 QAT direct update does not support {type(workspace).__name__}"
        )
    quantizer = workspace._quantizer
    # Build exactly the buffers this workspace holds. The quantizer's usage flags
    # are re-set every forward, so they can be narrower than the cached workspace;
    # deriving from the workspace keeps every buffer it owns consistent with the
    # weight instead of leaving a stale half behind.
    saved_usage = (quantizer.rowwise_usage, quantizer.columnwise_usage)
    quantizer.set_usage(
        rowwise=workspace._rowwise_data is not None,
        columnwise=workspace._columnwise_data is not None,
    )
    try:
        fresh = mxfp4_qat_direct_quantize(weight, quantizer)
    finally:
        quantizer.set_usage(rowwise=saved_usage[0], columnwise=saved_usage[1])
    if workspace._rowwise_data is not None:
        workspace._rowwise_data.copy_(fresh._rowwise_data)
        workspace._rowwise_scale_inv.copy_(fresh._rowwise_scale_inv)
    if workspace._columnwise_data is not None:
        # ``fresh`` already built its columnwise half above; rebuilding it here
        # would repeat the whole transpose on the blockwise path.
        workspace._columnwise_data.copy_(fresh._columnwise_data)
        workspace._columnwise_scale_inv.copy_(fresh._columnwise_scale_inv)
