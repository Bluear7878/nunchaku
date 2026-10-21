"""
NVFP4 SVDQuant layers on data-center Blackwell GPUs (sm_100 / sm_103, e.g. B200 / B300).

Nunchaku's NVFP4 GEMM uses ``mma.sync ... kind::mxf4nvf4``, which only sm_120 / sm_121 have. Data-center Blackwell
reaches its FP4 tensor cores through ``tcgen05.mma`` instead, so here the 4-bit GEMM runs in cuBLAS. Each
:class:`~nunchaku.models.linear.SVDQW4A4Linear` is converted once from Nunchaku's MMA layout to the layout cuBLAS
reads: by the model loaders right after loading, otherwise on first use.
"""

import functools

import torch
import torch.nn.functional as F

from .._C import ops
from ..lora.flux.packer import NunchakuWeightPacker
from ..utils import ceil_divide

_PACKER = NunchakuWeightPacker(bits=4, warp_n=128)

try:
    # e4m3 scales for every 16 values (swizzled) and one FP32 scale for the whole tensor
    _RECIPE = [F.ScalingType.BlockWise1x16.value, F.ScalingType.TensorWise.value]
    _SWIZZLE = [F.SwizzleType.SWIZZLE_32_4_4.value, F.SwizzleType.NO_SWIZZLE.value]
    _scaled_mm_v2 = torch._scaled_mm_v2
except AttributeError:  # torch < 2.10
    _RECIPE = _SWIZZLE = _scaled_mm_v2 = None


@functools.cache
def is_datacenter_blackwell(device: torch.device) -> bool:
    """Whether ``device`` is a data-center Blackwell GPU (compute capability 10.x, e.g. B200 and B300)."""
    return device.type == "cuda" and torch.cuda.get_device_capability(device)[0] == 10


def uses_nvfp4_sm100(layer, device: torch.device) -> bool:
    """Whether the SVDQW4A4Linear ``layer`` runs through this module for inputs on ``device``."""
    return layer.precision == "nvfp4" and is_datacenter_blackwell(device)


def pack_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit codes (..., k) into bytes (..., k // 2), the first code in the low nibble."""
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)


def to_blocked(scales: torch.Tensor) -> torch.Tensor:
    """
    Rearrange NVFP4 block scales of shape (rows, k // 16) into the swizzled layout cuBLAS reads.

    Rows are padded to a multiple of 128 and columns to a multiple of 4; each 128 x 4 tile is stored as 32 x 16.
    """
    rows, cols = scales.shape
    raw = F.pad(scales.view(torch.uint8), (0, -cols % 4, 0, -rows % 128))
    blocks = raw.reshape(raw.shape[0] // 128, 128, raw.shape[1] // 4, 4).permute(0, 2, 1, 3)
    blocked = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)
    return blocked.flatten().view(torch.float8_e4m3fn)


def quantize_activation(x: torch.Tensor, smooth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize activations to NVFP4 with the arithmetic of Nunchaku's kernel, in the layout cuBLAS reads.

    Parameters
    ----------
    x : torch.Tensor, shape (M, K), dtype bfloat16 or float16
        Input activations.
    smooth : torch.Tensor, shape (K,), same dtype as ``x``
        Smoothing factor; ``x`` is divided by it before quantization.

    Returns
    -------
    act : torch.Tensor, shape (M, K // 2), dtype float4_e2m1fn_x2
        Quantized activations, the first value in the low nibble.
    ascales : torch.Tensor, shape (ceil(M / 128) * 128 * K // 16,), dtype float8_e4m3fn
        One scale per 16 values, in the layout of :func:`to_blocked`.
    """
    m, k = x.shape
    act = torch.empty(m, k // 2, dtype=torch.uint8, device=x.device)
    ascales = torch.empty(ceil_divide(m, 128) * 128 * (k // 16), dtype=torch.float8_e4m3fn, device=x.device)
    ops.quantize_nvfp4_act_sm100(x, smooth, act, ascales)
    return act.view(torch.float4_e2m1fn_x2), ascales


@torch.no_grad()
def prepare_linear(layer) -> None:
    """
    Convert an NVFP4 SVDQW4A4Linear in place from Nunchaku's MMA layout to the layout used here.

    Converted layers hold ``qweight`` as uint8, which is how a second call skips them. Per-channel scales that are all
    1 are dropped (``wcscales`` becomes None). The state dict then holds this layout and must not be saved as a
    Nunchaku checkpoint.
    """
    if layer.qweight.dtype == torch.uint8:
        return
    n, k = layer.out_features, layer.in_features
    layer.qweight.data = pack_codes(_PACKER.unpack_weight(layer.qweight.data))
    layer.wscales.data = to_blocked(_PACKER.unpack_scale(layer.wscales.data, n, group_size=16))
    layer.smooth_factor.data = _PACKER.unpack_scale(layer.smooth_factor.data, k, group_size=-1).view(k)
    layer.proj_down.data = _PACKER.unpack_lowrank_weight(layer.proj_down.data, down=True).t().contiguous()
    layer.proj_up.data = _PACKER.unpack_lowrank_weight(layer.proj_up.data, down=False)
    if layer.bias is not None:
        layer.bias.data = _PACKER.unpack_scale(layer.bias.data, n, group_size=-1).view(n)
    if layer.wcscales is not None:
        if bool((layer.wcscales == 1).all()):
            layer.wcscales = None
        else:
            layer.wcscales.data = _PACKER.unpack_scale(layer.wcscales.data, n, group_size=-1).view(n)


def prepare_model(model: torch.nn.Module) -> None:
    """
    Convert the NVFP4 layers of ``model`` in place when the current GPU is a data-center Blackwell GPU.

    Model loaders call this right after loading the checkpoint, so that offloading set up later (CPU offload managers,
    accelerate hooks) stores the converted weights.
    """
    if not (torch.cuda.is_available() and is_datacenter_blackwell(torch.device("cuda"))):
        return
    from ..models.linear import SVDQW4A4Linear  # nunchaku.models.linear imports this module

    for module in model.modules():
        if isinstance(module, SVDQW4A4Linear) and module.precision == "nvfp4":
            prepare_linear(module)
        elif hasattr(module, "nvfp4_sm100_params"):  # holds the parameters of an SVDQW4A4Linear itself
            params = module.nvfp4_sm100_params()
            if params is not None:
                prepare_linear(params)


@functools.cache
def _scalar(value: float, device: torch.device) -> torch.Tensor:
    return torch.tensor(value, dtype=torch.float32, device=device)


def _gemm(act, ascales, layer, alpha: float, bias, dtype) -> torch.Tensor:
    """NVFP4 GEMM in cuBLAS; ``alpha`` and the bias are applied to the FP32 accumulator."""
    weight = layer.qweight.view(torch.float4_e2m1fn_x2).t()
    if _scaled_mm_v2 is None:
        out = torch._scaled_mm(act, weight, scale_a=ascales, scale_b=layer.wscales, out_dtype=torch.float32) * alpha
        return (out if bias is None else out + bias.float()).to(dtype)
    scales_a = [ascales, _scalar(1.0, act.device)]
    scales_b = [layer.wscales, _scalar(alpha, act.device)]
    return _scaled_mm_v2(act, weight, scales_a, _RECIPE, _SWIZZLE, scales_b, _RECIPE, _SWIZZLE, bias, dtype, [], False)


def linear_forward(layer, x: torch.Tensor) -> torch.Tensor:
    """
    Forward of an NVFP4 SVDQW4A4Linear: the low-rank branch plus the 4-bit GEMM on the smoothed input.

    Parameters
    ----------
    layer : SVDQW4A4Linear
        NVFP4 layer, converted by :func:`prepare_linear` if it has not been yet.
    x : torch.Tensor, shape (M, in_features), dtype bfloat16 or float16
        Input activations.

    Returns
    -------
    torch.Tensor, shape (M, out_features), same dtype as ``x``
        Output activations.
    """
    prepare_linear(layer)
    alpha = 1.0 if layer.wtscale is None else float(layer.wtscale)
    x = x.contiguous()
    lora_act = torch.mm(x, layer.proj_down)
    act, ascales = quantize_activation(x, layer.smooth_factor)
    if layer.wcscales is None:
        out = _gemm(act, ascales, layer, alpha, layer.bias, x.dtype)
    else:
        # cuBLAS has no per-channel scale on top of block scales; apply it with the bias in 16 bits, as EpilogueBias does
        out = _gemm(act, ascales, layer, alpha, None, x.dtype)
        ops.scale_bias_sm100(out, layer.wcscales, layer.bias)
    return out.addmm_(lora_act, layer.proj_up.t())


def gelu_mlp(x: torch.Tensor, fc1, fc2) -> torch.Tensor:
    """Two NVFP4 layers with tanh GELU in between, as :func:`~nunchaku.ops.fused.fused_gelu_mlp`."""
    batch_size, seq_len, channels = x.shape
    h = linear_forward(fc1, x.reshape(batch_size * seq_len, channels))
    h = F.gelu(h, approximate="tanh")
    return linear_forward(fc2, h).view(batch_size, seq_len, -1)


def qkv_norm_rope(x: torch.Tensor, proj, norm_q=None, norm_k=None, rotary_emb=None) -> torch.Tensor:
    """
    QKV projection followed by RMSNorm and rotary embedding on Q and K.

    Same as :func:`~nunchaku.ops.fused.fused_qkv_norm_rottary` with a single output tensor: the norms are applied
    together with ``rotary_emb`` only. ``norm_q`` and ``norm_k`` are RMSNorm modules or their weights; the epsilon is
    1e-6, as in the fused kernel.
    """
    batch_size, seq_len, channels = x.shape
    qkv = linear_forward(proj, x.reshape(batch_size * seq_len, channels))
    if rotary_emb is not None:
        weight_q = norm_q.weight if isinstance(norm_q, torch.nn.Module) else norm_q
        weight_k = norm_k.weight if isinstance(norm_k, torch.nn.Module) else norm_k
        ops.qk_rmsnorm_rope_sm100(qkv, weight_q, weight_k, rotary_emb, seq_len, 1e-6)
    return qkv.view(batch_size, seq_len, -1)


class LinearParams:
    """
    Parameters of an NVFP4 SVDQW4A4Linear held by another module (Z-Image's fused QKV), for :func:`linear_forward`.

    The attributes are that module's Parameter objects, so :func:`prepare_linear` converts them in place. Build it on
    every call: offloading hooks may replace the Parameter objects.
    """

    def __init__(self, qweight, wscales, smooth_factor, proj_down, proj_up, bias, wcscales, out_features, wtscale=None):
        self.qweight = qweight
        self.wscales = wscales
        self.smooth_factor = smooth_factor
        self.proj_down = proj_down
        self.proj_up = proj_up
        self.bias = bias
        self.wcscales = wcscales
        self.wtscale = wtscale
        self.in_features = qweight.shape[1] * 2
        self.out_features = out_features
