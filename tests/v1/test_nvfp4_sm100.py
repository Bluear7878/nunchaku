import pytest
import torch

from nunchaku.lora.flux.packer import NunchakuWeightPacker
from nunchaku.models.embeddings import pack_rotemb
from nunchaku.models.linear import SVDQW4A4Linear
from nunchaku.ops import nvfp4_sm100
from nunchaku.utils import copy_params_into

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and nvfp4_sm100.is_datacenter_blackwell(torch.device("cuda"))),
    reason="NVFP4 on sm_100 / sm_103 (B200, B300) only",
)

PACKER = NunchakuWeightPacker(bits=4, warp_n=128)
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def e2m1_codes(values: torch.Tensor) -> torch.Tensor:
    """Round to E2M1 codes with round-to-nearest-even and saturation, as ``cvt.rn.satfinite.e2m1x2`` does."""
    a = values.abs()
    code = torch.zeros_like(a, dtype=torch.uint8)
    code = torch.where(a > 0.25, 1, code)
    code = torch.where(a >= 0.75, 2, code)
    code = torch.where(a > 1.25, 3, code)
    code = torch.where(a >= 1.75, 4, code)
    code = torch.where(a > 2.5, 5, code)
    code = torch.where(a >= 3.5, 6, code)
    code = torch.where(a > 5.0, 7, code)
    return (code | torch.where(values < 0, 8, 0)).to(torch.uint8)


def e2m1_values(codes: torch.Tensor) -> torch.Tensor:
    magnitude = torch.tensor(E2M1_VALUES, device=codes.device)[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -magnitude, magnitude)


def quantize_reference(x: torch.Tensor, smooth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Nunchaku's NVFP4 activation quantization with exact division; returns (M, K) codes and (M, K / 16) scales."""
    m, k = x.shape
    xs = (x / smooth).float().view(m, k // 16, 16)
    scale = torch.clamp(xs.abs().amax(dim=-1) * torch.tensor(1 / 6), max=448.0)
    rscale = torch.where(scale > 0, 1.0 / scale, torch.zeros_like(scale))
    return e2m1_codes(xs * rscale.unsqueeze(-1)).view(m, k), scale.to(torch.float8_e4m3fn)


def dequantize(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    rows = codes.shape[0]
    return (e2m1_values(codes).view(rows, -1, 16) * scales.float().unsqueeze(-1)).view(rows, -1)


def test_unpacking_inverts_the_packer():
    n, k = 384, 512
    codes = torch.randint(0, 16, (n, k), dtype=torch.int32, device="cuda")
    assert torch.equal(PACKER.unpack_weight(PACKER.pack_weight(codes.clone())), codes)

    scales = torch.rand(n, k // 16, device="cuda").mul(448).to(torch.float8_e4m3fn)
    packed = PACKER.pack_scale(scales.to(torch.bfloat16), group_size=16)
    assert torch.equal(PACKER.unpack_scale(packed, n, group_size=16).view(torch.uint8), scales.view(torch.uint8))

    for length in (n, k):
        v = torch.randn(length, 1, device="cuda", dtype=torch.bfloat16)
        assert torch.equal(PACKER.unpack_scale(PACKER.pack_scale(v, group_size=-1), length, group_size=-1), v)


@pytest.mark.parametrize("m,k", [(300, 3072), (4608, 3072), (512, 12288)])
def test_activation_quantization_matches_reference(m: int, k: int):
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda") * torch.logspace(-4, 4, k, device="cuda")[torch.randperm(k, device="cuda")]
    x[:3, :64] = 0
    x = x.to(torch.bfloat16)
    smooth = (torch.rand(k, device="cuda") * 4 + 0.25).to(torch.bfloat16)

    act, ascales = nvfp4_sm100.quantize_activation(x, smooth)
    codes_ref, scales_ref = quantize_reference(x, smooth)
    assert torch.equal(ascales.view(torch.uint8), nvfp4_sm100.to_blocked(scales_ref).view(torch.uint8))
    raw = act.view(torch.uint8)
    codes = torch.stack((raw & 0xF, raw >> 4), dim=-1).view(m, k)
    # the kernel uses approximate division and reciprocal like Nunchaku's, so rare rounding ties may differ
    mismatch = (e2m1_values(codes) != e2m1_values(codes_ref)).float().mean().item()
    assert mismatch < 1e-4


@pytest.mark.parametrize("legacy_scaled_mm", [False, True])
def test_nvfp4_linear_matches_reference(legacy_scaled_mm: bool, monkeypatch):
    if legacy_scaled_mm:  # torch < 2.10 has no torch._scaled_mm_v2
        monkeypatch.setattr(nvfp4_sm100, "_scaled_mm_v2", None)
    torch.manual_seed(0)
    m, k, n, rank = 300, 512, 384, 32
    codes = torch.randint(0, 16, (n, k), dtype=torch.int32, device="cuda")
    wscales = torch.rand(n, k // 16, device="cuda").mul(2).to(torch.float8_e4m3fn)
    smooth = (torch.rand(k, device="cuda") + 0.5).to(torch.bfloat16)
    down = torch.randn(rank, k, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(n, rank, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    wcscales = (torch.rand(n, device="cuda") + 0.5).to(torch.bfloat16)

    for use_wcscales, alpha in ((False, 0.75), (True, 1.0)):
        layer = SVDQW4A4Linear(k, n, rank=rank, bias=True, precision="nvfp4", torch_dtype=torch.bfloat16, device="cuda")
        layer.qweight.data.copy_(PACKER.pack_weight(codes.clone()))
        layer.wscales.data.copy_(PACKER.pack_micro_scale(wscales.to(torch.bfloat16), group_size=16))
        layer.smooth_factor.data.copy_(PACKER.pack_scale(smooth.view(k, 1), group_size=-1))
        layer.proj_down.data.copy_(PACKER.pack_lowrank_weight(down, down=True))
        layer.proj_up.data.copy_(PACKER.pack_lowrank_weight(up, down=False))
        layer.bias.data.copy_(PACKER.pack_scale(bias.view(n, 1), group_size=-1))
        channel_scales = wcscales if use_wcscales else torch.ones_like(wcscales)
        layer.wcscales.data.copy_(PACKER.pack_scale(channel_scales.view(n, 1), group_size=-1))
        layer.wtscale = alpha

        x = torch.randn(1, m, k, device="cuda", dtype=torch.bfloat16)
        y = layer(x).float().view(m, n)

        act_codes, act_scales = quantize_reference(x.view(m, k), smooth)
        main = dequantize(act_codes, act_scales) @ dequantize(codes.to(torch.uint8), wscales).t() * alpha
        main = main.to(torch.bfloat16).float()
        if use_wcscales:
            main = (main * wcscales.float()).to(torch.bfloat16).float()
        expected = main + bias.float() + (x.view(m, k) @ down.t()).float() @ up.float().t()
        assert ((y - expected).norm() / expected.norm()).item() < 1e-2


def test_offload_buffer_runs_the_copied_layer():
    # CPUOffloadManager copies each block's converted parameters and wtscale into a resident buffer block
    torch.manual_seed(0)
    k, n, rank = 512, 384, 32
    layers = []
    for wtscale in (0.5, 2.0):
        layer = SVDQW4A4Linear(k, n, rank=rank, bias=True, precision="nvfp4", torch_dtype=torch.bfloat16, device="cuda")
        layer.qweight.data.copy_(PACKER.pack_weight(torch.randint(0, 16, (n, k), dtype=torch.int32, device="cuda")))
        wscales = (torch.rand(n, k // 16, device="cuda") + 0.5).to(torch.bfloat16)
        layer.wscales.data.copy_(PACKER.pack_micro_scale(wscales, group_size=16))
        for p in (layer.smooth_factor, layer.wcscales):
            p.data.copy_(torch.rand_like(p) + 0.5)
        for p in (layer.proj_down, layer.proj_up, layer.bias):
            p.data.normal_(0, 0.1)
        layer.wtscale = wtscale
        nvfp4_sm100.prepare_model(layer)
        layers.append(layer)
    source, buffer = layers
    x = torch.randn(1, 300, k, device="cuda", dtype=torch.bfloat16)
    expected = source(x)
    copy_params_into(source, buffer)
    assert torch.equal(buffer(x), expected)


def test_qk_rmsnorm_rope_matches_reference():
    torch.manual_seed(0)
    batch, seq, heads = 2, 512, 4
    qkv = torch.randn(batch * seq, 3 * heads * 128, device="cuda").mul(3).to(torch.bfloat16)
    norm_q = (torch.rand(128, device="cuda") + 0.5).to(torch.bfloat16)
    norm_k = (torch.rand(128, device="cuda") + 0.5).to(torch.bfloat16)
    angle = torch.rand(1, seq, 64, device="cuda") * 6.28
    rotary = pack_rotemb(torch.stack((angle.sin(), angle.cos()), dim=-1).view(1, seq, 64, 1, 2))

    q, k, v = qkv.float().view(batch, seq, 3, heads, 128).unbind(2)
    sin, cos = angle.sin().unsqueeze(2), angle.cos().unsqueeze(2)

    def transform(t, w):
        t = t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
        t0, t1 = t[..., 0::2], t[..., 1::2]
        return torch.stack((t0 * cos - t1 * sin, t0 * sin + t1 * cos), -1).flatten(-2)

    expected = torch.stack((transform(q, norm_q), transform(k, norm_k), v), dim=2).view(batch * seq, -1)
    out = qkv.clone()
    nvfp4_sm100.ops.qk_rmsnorm_rope_sm100(out, norm_q, norm_k, rotary, seq, 1e-6)
    assert ((out.float() - expected).norm() / expected.norm()).item() < 5e-3
