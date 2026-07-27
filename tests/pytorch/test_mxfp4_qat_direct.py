# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for the direct MXFP4 -> FP8 converter (TileKernels canonicalization).

Contracts:
  - rowwise MXFP8 bytes follow the deployment fixed-shift-6 lift (invertible,
    scale code = fp4_exp + 121 clamped at 0 with exact payload absorption);
  - decoded values equal the bridge path (mxfp4_fake_quantize) bit for bit;
  - columnwise MXFP8 bytes equal the bridge quantizer's (host amax rule);
  - blockwise 128x128 uses tile scale 2^(max fp4 exp - 6) with RTNE folding:
    exact through spread 2^14, bounded beyond, no device assert;
  - end to end, direct and bridge recipes produce bitwise-identical outputs
    and gradients (equivalent FP8 factorizations through the real GEMMs).
"""
import ctypes
import os

for _n in ("libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so"):
    try:
        ctypes.CDLL(_n, mode=ctypes.RTLD_GLOBAL)
        break
    except OSError:
        continue

import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common.recipe import (
    MXFP4QATFloat8BlockScaling,
    MXFP4QATMXFP8BlockScaling,
)
from transformer_engine.pytorch import fp8_autocast
from transformer_engine.pytorch.mxfp4_qat import mxfp4_fake_quantize
from transformer_engine.pytorch.mxfp4_qat_direct import mxfp4_qat_direct_quantize
from references.mxfp4_qat_direct_reference import (
    _direct_blockwise_128,
    _direct_mxfp8_rowwise,
    _mxfp4_decompose,
)
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

torch.manual_seed(4321)
DEV = "cuda"
RUN_PERF = os.environ.get("NVTE_MXFP4_QAT_PERF", "0") == "1"


def make_weight(m, n, dtype=torch.bfloat16, zero_blocks=0, outliers=0):
    w = (torch.randn(m, n, device=DEV, dtype=torch.float32) * 0.02).to(dtype)
    if outliers:
        idx = torch.randint(0, m * n, (outliers,), device=DEV)
        w.view(-1)[idx] *= 500.0
    if zero_blocks:
        for _ in range(zero_blocks):
            r = torch.randint(0, m, (1,)).item()
            k = torch.randint(0, n // 32, (1,)).item()
            w[r, k * 32 : (k + 1) * 32] = 0.0
    return w


def _decode_mxfp8_rowwise(qt, m, n):
    data = qt._rowwise_data.view(torch.uint8)[:m, :n]
    codes = qt._rowwise_scale_inv.view(torch.uint8)[:m, : n // 32].to(torch.int32)
    vals = data.view(torch.float8_e4m3fn).to(torch.float32).view(m, n // 32, 32)
    scale = torch.ldexp(torch.ones_like(codes, dtype=torch.float32), codes - 127)
    return (vals * scale.unsqueeze(-1)).view(m, n)


def _decode_blockwise_rowwise(qt, m, n):
    data = qt._rowwise_data.view(torch.uint8)[:m, :n]
    scales = qt._rowwise_scale_inv[: m // 128, : n // 128]
    vals = data.view(torch.float8_e4m3fn).to(torch.float32)
    up = scales.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return vals * up


def test_rowwise_canonical_bytes_and_decode():
    m, n = 64, 256
    w = make_weight(m, n, zero_blocks=3, outliers=3)
    w[0, :32] = torch.tensor(2.0**-127, dtype=torch.bfloat16)  # floor-scale block
    w[1, :32] = 0.0
    w[1, 0] = -0.0
    w[2, :32] = 0.0
    w[2, 0] = 3.25  # quantizes to payload 3 at fp4 exp 0 -> canonical divergence case
    w[3, :32] = 0.0
    w[3, 0] = 6.0  # payload 6 at fp4 exp 0 -> canonical agreement case

    q = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, columnwise=False)
    qt = mxfp4_qat_direct_quantize(w, q)
    what = mxfp4_fake_quantize(w)

    # decoded values == bridge values, bitwise, everywhere (incl. 2^-127)
    dec = _decode_mxfp8_rowwise(qt, m, n)
    assert torch.equal(dec, what.to(torch.float32)), "direct decode != bridge values"

    data = qt._rowwise_data.view(torch.uint8)
    codes = qt._rowwise_scale_inv.view(torch.uint8)

    # fixed-shift canonicalization: scale code = e + 121 (recomputed independently)
    e, _, _ = _mxfp4_decompose(w)
    expect_codes = (e.view(m, n // 32) + 121).clamp(min=0).to(torch.uint8)
    assert torch.equal(codes[:m, : n // 32], expect_codes), "scale codes not fixed-shift"

    # pmax=3 block: TileKernels bytes (192 @ 2^-6), differing from the bridge's
    assert int(codes[2, 0]) == 121 and int(data[2, 0]) == 0x74, "pmax=3 canonical bytes"
    bridge_qt = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, columnwise=False).quantize(what)
    bdata = bridge_qt._rowwise_data.view(torch.uint8)
    bcodes = bridge_qt._rowwise_scale_inv.view(torch.uint8)
    assert (int(bcodes[2, 0]), int(bdata[2, 0])) == (120, 0x7C), "bridge canonical changed?"
    # pmax=6 block: the two canonicalizations coincide
    assert int(codes[3, 0]) == int(bcodes[3, 0]) and int(data[3, 0]) == int(bdata[3, 0])

    # floor block: code 0, payload absorbs (0.5 * 2^(e+127) = 1.0 -> byte 56)
    assert int(codes[0, 0]) == 0 and int(data[0, 0]) == 56, "code-0 absorption"
    # zero block: TileKernels clamp convention -> code 0, payload 0; -0 keeps sign
    assert int(codes[1, 0]) == 0 and int(data[1, 1]) == 0
    assert int(data[1, 0]) == 0x80, "-0 payload sign lost"
    print("  rowwise canonical bytes + decode: PASS")


def test_colwise_matches_bridge():
    m, n = 128, 256
    w = make_weight(m, n, zero_blocks=4, outliers=4)
    what = mxfp4_fake_quantize(w)

    qd = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    dt = mxfp4_qat_direct_quantize(w, qd)
    qb = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    bt = qb.quantize(what)

    dcol = dt._columnwise_data.view(torch.uint8)
    bcol = bt._columnwise_data.view(torch.uint8)
    assert torch.equal(dcol, bcol), "columnwise payload bytes differ from bridge"
    dsc = dt._columnwise_scale_inv.view(torch.uint8)[: m // 32, :n]
    bsc = bt._columnwise_scale_inv.view(torch.uint8)[: m // 32, :n]
    # zero columns may legitimately carry different (unused) scale codes
    nz = dcol.view(m // 32, 32, n).ne(0).any(dim=1)
    assert torch.equal(dsc[nz], bsc[nz]), "columnwise scale codes differ from bridge"
    print("  colwise bytes == bridge: PASS")


def test_blockwise_folding_and_decode():
    def mk_quantizer():
        return Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True,
            force_pow_2_scales=True, block_scaling_dim=2,
        )

    # spread sweep: exact through d<=14, first loss at d=15 (E4M3 subnormal bound)
    payloads = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    boundary = None
    for d in range(17):
        w = torch.zeros(128, 128, dtype=torch.float32, device=DEV)
        w[0, 0] = 6.0
        w[1, :7] = payloads.to(DEV) * 2.0**-d
        w[1, 7:14] = -w[1, :7]
        qt = mxfp4_qat_direct_quantize(w, mk_quantizer())
        assert qt._rowwise_scale_inv[0, 0].item() == 2.0 ** (0 - 6), f"tile scale d={d}"
        dec = _decode_blockwise_rowwise(qt, 128, 128)
        exact = torch.equal(dec, mxfp4_fake_quantize(w).to(torch.float32))
        if boundary is None and not exact:
            boundary = d
    assert boundary == 15, f"expected first loss at spread 15, got {boundary}"

    # normal weights: decode + TE dequantize + columnwise transpose all exact
    w = make_weight(256, 256, zero_blocks=4).to(torch.bfloat16)
    what = mxfp4_fake_quantize(w)
    qt = mxfp4_qat_direct_quantize(w, mk_quantizer())
    assert torch.equal(_decode_blockwise_rowwise(qt, 256, 256), what.to(torch.float32))
    assert torch.equal(qt.dequantize(dtype=torch.float32), what.to(torch.float32))
    qt.update_usage(rowwise_usage=False, columnwise_usage=True)
    assert torch.equal(qt.dequantize(dtype=torch.float32), what.to(torch.float32))

    # all-zero tile: clamped scale, zero payloads
    wz = torch.zeros(128, 128, dtype=torch.bfloat16, device=DEV)
    qz = mxfp4_qat_direct_quantize(wz, mk_quantizer())
    assert (qz._rowwise_data.view(torch.uint8) == 0).all()
    assert qz._rowwise_scale_inv[0, 0].item() == 2.0**-127
    print(f"  blockwise folding: exact through d=14, first loss at d={boundary}: PASS")


def test_e2e_direct_matches_bridge():
    torch.manual_seed(7)
    tokens, in_f, out_f = 192, 256, 256
    x0 = torch.randn(tokens, in_f, device=DEV, dtype=torch.bfloat16)

    for recipe_cls in (MXFP4QATMXFP8BlockScaling, MXFP4QATFloat8BlockScaling):
        for override in (None, "dequantized", "high_precision"):
            results = {}
            for direct in (False, True):
                torch.manual_seed(11)
                mod = te.Linear(in_f, out_f, bias=False, params_dtype=torch.bfloat16, device=DEV)
                recipe = recipe_cls()
                recipe.backward_override = override
                recipe.mxfp4_qat_direct = direct
                x = x0.clone().requires_grad_(True)
                outs = []
                for first in (True, False, True):  # miss, cache-hit, update paths
                    with fp8_autocast(enabled=True, fp8_recipe=recipe):
                        out = mod(x, is_first_microbatch=first)
                    outs.append(out.detach().clone())
                out.backward(torch.ones_like(out))
                results[direct] = (
                    outs,
                    x.grad.detach().clone(),
                    mod.weight.grad.detach().clone(),
                )
            for i, (ob, od) in enumerate(zip(results[False][0], results[True][0])):
                assert torch.equal(ob, od), (
                    f"fwd differs (bridge vs direct) {recipe_cls.__name__} "
                    f"override={override} step={i}"
                )
            assert torch.equal(results[False][1], results[True][1]), (
                f"dgrad differs {recipe_cls.__name__} override={override}"
            )
            assert torch.equal(results[False][2], results[True][2]), (
                f"wgrad differs {recipe_cls.__name__} override={override}"
            )
    print("  e2e direct == bridge (fwd/dgrad/wgrad bitwise, 2 recipes x 3 overrides): PASS")


def test_direct_flag_plumbing():
    r = MXFP4QATMXFP8BlockScaling()
    assert r.mxfp4_qat() and not r.mxfp4_qat_direct_conversion()
    r.mxfp4_qat_direct = True
    assert r.mxfp4_qat_direct_conversion()
    from transformer_engine.common.recipe import MXFP8BlockScaling

    base = MXFP8BlockScaling()
    base.mxfp4_qat_direct = True  # direct without QAT is inert
    assert not base.mxfp4_qat_direct_conversion()
    print("  recipe flag plumbing: PASS")



def _edge_weight():
    w = make_weight(128, 256)
    w[0, :32] = torch.tensor(2.0**-127, dtype=torch.bfloat16)
    w[1, :32] = 0.0
    w[1, 0] = -0.0
    w[2, :32] = 0.0
    w[2, 0] = 3.25
    w[3, :32] = torch.tensor(2.0**120, dtype=torch.bfloat16)
    w[4, 5] = float("inf")
    w[5, 9] = float("nan")
    return w


_DIRECT_KERNELS = {}


def _intree_direct(fast_math=False):
    """JIT-compile the in-tree CUDA direct kernels for testing."""
    if fast_math not in _DIRECT_KERNELS:
        from torch.utils.cpp_extension import load_inline

        src_dir = os.environ.get(
            "NVTE_MXFP4_QAT_TEST_SRC",
            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "..", "transformer_engine", "common"),
        )
        cuda_src = r"""
#include <cuda.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "cast/mxfp4/direct_convert_mxfp4.cuh"

std::vector<torch::Tensor> direct_row(torch::Tensor w) {
  TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.dim() == 2 && w.size(1) % 32 == 0);
  const int64_t rows = w.size(0), cols = w.size(1), bpr = cols / 32;
  auto data = torch::empty({rows, cols}, w.options().dtype(torch::kUInt8));
  auto scales = torch::zeros({(rows + 127) / 128 * 128, (bpr + 3) / 4 * 4},
                             w.options().dtype(torch::kUInt8));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int ss = scales.size(1);
  if (w.scalar_type() == at::kBFloat16) {
    transformer_engine::dispatch::mxfp4::direct_mxfp8_rowwise_launch<nv_bfloat16>(
        reinterpret_cast<const nv_bfloat16 *>(w.data_ptr()), data.data_ptr<uint8_t>(),
        scales.data_ptr<uint8_t>(), rows, cols, ss, stream);
  } else {
    transformer_engine::dispatch::mxfp4::direct_mxfp8_rowwise_launch<float>(
        w.data_ptr<float>(), data.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(), rows, cols,
        ss, stream);
  }
  return {data, scales};
}

std::vector<torch::Tensor> direct_block(torch::Tensor w) {
  TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.dim() == 2);
  TORCH_CHECK(w.size(0) % 128 == 0 && w.size(1) % 128 == 0);
  const int64_t rows = w.size(0), cols = w.size(1), tm = rows / 128, tn = cols / 128;
  auto data = torch::empty({rows, cols}, w.options().dtype(torch::kUInt8));
  auto scales = torch::zeros({tm, (tn + 3) / 4 * 4}, w.options().dtype(torch::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  const int ss = scales.size(1);
  if (w.scalar_type() == at::kBFloat16) {
    transformer_engine::dispatch::mxfp4::direct_blockwise_launch<nv_bfloat16>(
        reinterpret_cast<const nv_bfloat16 *>(w.data_ptr()), data.data_ptr<uint8_t>(),
        scales.data_ptr<float>(), rows, cols, ss, stream);
  } else {
    transformer_engine::dispatch::mxfp4::direct_blockwise_launch<float>(
        w.data_ptr<float>(), data.data_ptr<uint8_t>(), scales.data_ptr<float>(), rows, cols, ss,
        stream);
  }
  return {data, scales};
}
"""
        _DIRECT_KERNELS[fast_math] = load_inline(
            name="nvte_mxfp4_direct_intree" + ("_fastmath" if fast_math else ""),
            cpp_sources=(
                "#include <vector>\n#include <torch/extension.h>\n"
                "std::vector<torch::Tensor> direct_row(torch::Tensor w);\n"
                "std::vector<torch::Tensor> direct_block(torch::Tensor w);"
            ),
            cuda_sources=cuda_src,
            functions=["direct_row", "direct_block"],
            extra_include_paths=[src_dir],
            extra_cuda_cflags=[
                "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            ] + (["--use_fast_math"] if fast_math else []),
            verbose=False,
        )
    return _DIRECT_KERNELS[fast_math]


def test_cuda_kernels_parity():
    """In-tree CUDA direct kernels: bitwise parity, normal and fast-math builds."""
    for fast_math in (False, True):
        k = _intree_direct(fast_math=fast_math)
        tag = "fast-math" if fast_math else "normal"
        for m, n in [(1, 32), (3, 416), (128, 256), (192, 448)]:
            for dtype in (torch.bfloat16, torch.float32):
                w = make_weight(m, n, dtype=dtype, zero_blocks=2, outliers=2)
                dd, ds = k.direct_row(w.contiguous())
                td, ts = _direct_mxfp8_rowwise(*_mxfp4_decompose(w), m, n)
                assert torch.equal(dd, td) and torch.equal(ds, ts), (
                    f"cuda row {tag} differs {m}x{n} {dtype}"
                )
        wx = _edge_weight()
        dd, ds = k.direct_row(wx.contiguous())
        td, ts = _direct_mxfp8_rowwise(*_mxfp4_decompose(wx), 128, 256)
        assert torch.equal(dd, td) and torch.equal(ds, ts), f"cuda row {tag} edge differs"

        for m, n in [(128, 128), (256, 384)]:
            for dtype in (torch.bfloat16, torch.float32):
                w = make_weight(m, n, dtype=dtype, zero_blocks=3, outliers=3)
                dd, ds = k.direct_block(w.contiguous())
                td, ts = _direct_blockwise_128(*_mxfp4_decompose(w), m, n)
                assert torch.equal(dd, td) and torch.equal(ds, ts), (
                    f"cuda blockwise {tag} differs {m}x{n} {dtype}"
                )
        payloads = torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        for d in (0, 11, 14, 15):
            w = torch.zeros(128, 128, dtype=torch.float32, device=DEV)
            w[0, 0] = 6.0
            w[1, :7] = payloads.to(DEV) * 2.0**-d
            dd, ds = k.direct_block(w)
            td, ts = _direct_blockwise_128(*_mxfp4_decompose(w), 128, 128)
            assert torch.equal(dd, td) and torch.equal(ds, ts), f"cuda blockwise {tag} d={d}"
        wx2 = _edge_weight()
        dd, ds = k.direct_block(wx2)
        td, ts = _direct_blockwise_128(*_mxfp4_decompose(wx2), 128, 256)
        assert torch.equal(dd, td) and torch.equal(ds, ts), f"cuda blockwise {tag} edge differs"

    if RUN_PERF:
        k = _intree_direct()
        rows = cols = 8192
        wb = make_weight(rows, cols)
        for _ in range(3):
            k.direct_row(wb)
            k.direct_block(wb)
        torch.cuda.synchronize()

        def timeit(fn, iters=20):
            s0, e0 = torch.cuda.Event(True), torch.cuda.Event(True)
            s0.record()
            for _ in range(iters):
                fn()
            e0.record()
            torch.cuda.synchronize()
            return s0.elapsed_time(e0) / iters * 1e3

        t_row = timeit(lambda: k.direct_row(wb))
        t_blk = timeit(lambda: k.direct_block(wb))
        moved = rows * cols * 2 + rows * cols + rows * cols // 32
        print(
            f"  cuda direct row: {t_row:.1f} us ({moved / t_row / 1e3:.0f} GB/s), "
            f"blockwise: {t_blk:.1f} us"
        )
    print("  cuda kernels (normal + fast-math): PASS")



TESTS = [
    test_rowwise_canonical_bytes_and_decode,
    test_cuda_kernels_parity,
    test_colwise_matches_bridge,
    test_blockwise_folding_and_decode,
    test_direct_flag_plumbing,
    test_e2e_direct_matches_bridge,
]

if __name__ == "__main__":
    import sys
    import traceback

    failed = 0
    for t in TESTS:
        try:
            t()
            torch.cuda.synchronize()
            print(f"PASS {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
