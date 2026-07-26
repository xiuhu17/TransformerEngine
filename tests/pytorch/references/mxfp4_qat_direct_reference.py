# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pure-PyTorch reference for the direct MXFP4 -> FP8 weight converters.

Bit-identical specification of tex.mxfp4_direct_mxfp8_rowwise (TileKernels
fixed-shift-6 canonicalization: scale code = fp4_exp + 121 clamped at UE8M0
code 0 with exact payload absorption) and tex.mxfp4_direct_blockwise
(per-tile exponent folding: tile scale = 2^(max block exponent - 6) clamped
at 2^-127, payloads RTNE onto E4M3 with subnormals -- exact through scale
spread 2^14, bounded beyond, no device assert). All-zero blocks pin the fp4
exponent at -126 (TileKernels clamp); non-finite 1x32 blocks emit E4M3 NaN
payloads (0x7F).
"""
import torch

_E4M3_NAN = 0x7F
_MXFP4_BLOCK = 32


def _roundup(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _round_to_e2m1_grid(y):
    """RTNE onto the E2M1 magnitude grid {0, .5, 1, 1.5, 2, 3, 4, 6}; input in [0, 6]."""
    fine = torch.round(y * 2.0) * 0.5
    mid = torch.round(y)
    coarse = torch.round(y * 0.5) * 2.0
    return torch.where(y <= 2.0, fine, torch.where(y <= 4.0, mid, coarse))


def _mxfp4_decompose(weight: torch.Tensor):
    """Per-1x32-block MXFP4 decomposition of a bf16/fp32 weight.

    Returns (e, q, nonfinite): block scale exponents e (int32, [R, B, 1],
    zero blocks pinned at -126 per the TileKernels clamp), signed E2M1 payload
    values q (fp32, [R, B, 32]), and the non-finite block mask ([R, B, 1]).
    """
    rows, cols = weight.shape
    if cols % _MXFP4_BLOCK != 0:
        raise ValueError(f"inner dim must be divisible by {_MXFP4_BLOCK}, got {cols}")
    w32 = weight.contiguous().to(torch.float32).view(rows, cols // _MXFP4_BLOCK, _MXFP4_BLOCK)
    amax = w32.abs().amax(dim=-1, keepdim=True)
    nonfinite = ~torch.isfinite(amax)

    bits = amax.view(torch.int32)
    exp_field = bits >> 23
    mantissa = bits & 0x7FFFFF
    e = exp_field - 129 + (mantissa > 0x400000).to(torch.int32)
    e = torch.where(exp_field > 0, e, torch.full_like(e, -126))
    e = e.clamp(min=-126, max=125)
    # all-zero block: TileKernels clamps amax at 6*2^-126 -> fp4 exponent -126
    # (deployment-canonical; payload 0 regardless, so the value is exact zero)
    e = torch.where(amax > 0, e, torch.full_like(e, -126))
    e = torch.where(nonfinite, torch.full_like(e, -126), e)

    scale = torch.ldexp(torch.ones_like(amax), e)
    y = (w32 / scale).abs().clamp(max=6.0)
    q = _round_to_e2m1_grid(torch.where(nonfinite, torch.zeros_like(y), y))
    q = torch.copysign(q, w32)
    return e, q, nonfinite


def _encode_e4m3(values: torch.Tensor) -> torch.Tensor:
    """RTNE encode fp32 values into E4M3 payload bytes (uint8)."""
    return values.to(torch.float8_e4m3fn).view(torch.uint8)


def _direct_mxfp8_rowwise(e, q, nonfinite, rows, cols):
    """TileKernels fixed-shift-6 lift: payload p*2^6 at scale code e+121."""
    ones = torch.ones_like(e, dtype=torch.float32)
    biased = e + 121  # e - 6 + 127
    code = biased.clamp(min=0)
    # below UE8M0 code 0 the scale pins at 2^-127 and the payload absorbs the
    # rest: p * 2^(e+127) (e in [-126, -122] -> multiplier 2..32, exact)
    shift = torch.where(biased >= 0, torch.full_like(e, 6), e + 127)
    payload_vals = q * torch.ldexp(ones, shift)
    data = _encode_e4m3(payload_vals)
    data = torch.where(
        nonfinite.expand_as(data), torch.full_like(data, _E4M3_NAN), data
    ).view(rows, cols)
    code = torch.where(nonfinite, torch.full_like(code, 127), code)

    scale_inv = torch.zeros(
        (_roundup(rows, 128), _roundup(cols // _MXFP4_BLOCK, 4)),
        dtype=torch.uint8,
        device=data.device,
    )
    scale_inv[:rows, : cols // _MXFP4_BLOCK] = code.view(rows, -1).to(torch.uint8)
    return data, scale_inv


def _host_amax_scale_exp(amax: torch.Tensor) -> torch.Tensor:
    """Host MXFP8 encoder scale rule: 2^ceil(log2(amax/448)), floored at code 0.

    ceil(log2(m*2^k / 448)) = k - 8 + (m > 1.75), computed from the bits.
    """
    bits = amax.view(torch.int32)
    exp_field = bits >> 23
    mantissa = bits & 0x7FFFFF
    k = exp_field - 127
    ec = k - 8 + (mantissa > 0x600000).to(torch.int32)  # 1.75 mantissa = 0x600000
    ec = torch.where(exp_field > 0, ec, torch.full_like(ec, -127))  # subnormal amax
    ec = ec.clamp(min=-127, max=127)
    ec = torch.where(amax > 0, ec, torch.zeros_like(ec))  # zero block: scale 1
    return ec


def _direct_mxfp8_columnwise(e, q, nonfinite, rows, cols):
    """Fresh 32x1 quantization of the grid values with the host amax rule.

    Matches the bridge path's columnwise bytes (same scale rule, same RTNE
    lattice) so mode-None dgrad is bit-identical across the two converters.
    """
    ones = torch.ones_like(e, dtype=torch.float32)
    vals = (q * torch.ldexp(ones, e)).view(rows, cols)
    nf_elem = nonfinite.expand(-1, -1, _MXFP4_BLOCK).reshape(rows, cols)

    vc = vals.view(rows // 32, 32, cols)
    amax_c = vc.abs().amax(dim=1, keepdim=True)
    finite_c = torch.isfinite(amax_c)
    amax_c = torch.where(finite_c, amax_c, torch.zeros_like(amax_c))
    ec = _host_amax_scale_exp(amax_c)
    inv = torch.ldexp(torch.ones_like(ec, dtype=torch.float32), -ec)
    # + 0.0 canonicalizes -0 to +0: the host colwise kernel drops the sign of
    # zero (its rowwise kernel keeps it) and byte-parity with the bridge is the
    # contract here; the value is unchanged either way.
    payload = _encode_e4m3(vc * inv + 0.0).view(rows, cols)
    payload = torch.where(nf_elem, torch.full_like(payload, _E4M3_NAN), payload)

    code = (ec + 127).clamp(0, 254).to(torch.uint8)
    scale_inv = torch.zeros(
        (_roundup(rows // 32, 4), _roundup(cols, 128)),
        dtype=torch.uint8,
        device=payload.device,
    )
    scale_inv[: rows // 32, :cols] = code.view(rows // 32, cols)
    return payload, scale_inv


def _direct_blockwise_128(e, q, nonfinite, rows, cols):
    """per_block_cast_lossless exponent folding, RTNE onto E4M3 (no assert).

    Tile scale = 2^(max block exponent - 6), clamped to >= 2^-127 so the
    fp32-stored scale stays on the E8M0 lattice; folded payloads round RTNE
    (E4M3 subnormals included): exact through spread 2^14, bounded beyond.
    """
    if rows % 128 != 0 or cols % 128 != 0:
        raise ValueError(f"blockwise direct conversion needs 128-divisible dims, got {rows}x{cols}")
    tiles_m, tiles_n = rows // 128, cols // 128

    e_eff = torch.where(nonfinite, torch.full_like(e, -126), e)
    # [R, B] -> [Tm, 128, Tn, 4] -> max over the 512 blocks of each tile
    e_t = e_eff.view(rows, cols // _MXFP4_BLOCK).view(tiles_m, 128, tiles_n, 128 // _MXFP4_BLOCK)
    max_e = e_t.amax(dim=(1, 3), keepdim=True)
    s_exp = (max_e - 6).clamp(min=-127)

    fold = (e_t - s_exp).view(rows, cols // _MXFP4_BLOCK, 1).to(torch.float32)
    payload_vals = q * torch.ldexp(torch.ones_like(fold), fold.to(torch.int32))
    data = _encode_e4m3(payload_vals)
    data = torch.where(
        nonfinite.expand_as(data), torch.full_like(data, _E4M3_NAN), data
    ).view(rows, cols)

    s_exp2 = s_exp.view(tiles_m, tiles_n)
    scale_inv = torch.zeros(
        (tiles_m, _roundup(tiles_n, 4)), dtype=torch.float32, device=data.device
    )
    scale_inv[:, :tiles_n] = torch.ldexp(
        torch.ones_like(s_exp2, dtype=torch.float32), s_exp2
    )
    return data, scale_inv


def mxfp4_direct_mxfp8_rowwise_reference(weight: torch.Tensor):
    """(payload bytes (M, N) uint8, padded UE8M0 scale codes) for a weight."""
    rows, cols = weight.shape
    return _direct_mxfp8_rowwise(*_mxfp4_decompose(weight), rows, cols)


def mxfp4_direct_blockwise_reference(weight: torch.Tensor):
    """(payload bytes (M, N) uint8, padded fp32 tile scales) for a weight."""
    rows, cols = weight.shape
    return _direct_blockwise_128(*_mxfp4_decompose(weight), rows, cols)
