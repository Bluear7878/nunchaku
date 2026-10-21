"""
This module provides the Nunchaku version of the FLUX.3 transformer (``Flux3Transformer2DModel`` in diffusers).
"""

import itertools
import json
import os
from pathlib import Path

from diffusers import Flux3Transformer2DModel
from huggingface_hub import utils

from ...utils import get_precision
from ..attention import _patch_linear
from ..linear import SVDQW4A4Linear
from ..utils import fuse_linears
from .utils import NunchakuModelLoaderMixin, patch_scale_key


class NunchakuFlux3Transformer2DModel(Flux3Transformer2DModel, NunchakuModelLoaderMixin):
    """
    Nunchaku-optimized Flux3Transformer2DModel.

    In every per-mode and joint transformer block, the attention and feed-forward projections are
    :class:`~nunchaku.models.linear.SVDQW4A4Linear` layers, with the query, key and value projections fused into one
    ``to_qkv``. QK normalization, RoPE and attention run in diffusers' ``Flux3AttnProcessor``.
    """

    def _patch_model(self, **kwargs):
        """
        Replace the attention and feed-forward projections of every transformer block with quantized linears.

        Parameters
        ----------
        **kwargs
            Additional arguments for quantization (e.g., precision, rank).

        Returns
        -------
        self : NunchakuFlux3Transformer2DModel
            The patched model.
        """
        for block in itertools.chain(self.txt_mode_blocks, *self.mode_blocks.values(), self.single_transformer_blocks):
            attn = block.attn
            attn.to_qkv = SVDQW4A4Linear.from_linear(fuse_linears([attn.to_q, attn.to_k, attn.to_v]), **kwargs)
            del attn.to_q, attn.to_k, attn.to_v
            attn.fused_projections = True
            # unfuse_projections() would delete the quantized to_qkv
            attn._supports_qkv_fusion = False
            attn.to_out[0] = SVDQW4A4Linear.from_linear(attn.to_out[0], **kwargs)
            _patch_linear(block.ff, SVDQW4A4Linear, **kwargs)
        return self

    @classmethod
    @utils.validate_hf_hub_args
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike[str], **kwargs):
        """
        Load a pretrained NunchakuFlux3Transformer2DModel from a safetensors file.

        Parameters
        ----------
        pretrained_model_name_or_path : str or os.PathLike
            Path to the safetensors file. It can be a local file or a remote HuggingFace path.
        **kwargs
            Additional arguments (e.g., device, torch_dtype).

        Returns
        -------
        NunchakuFlux3Transformer2DModel
            The loaded and quantized model.

        Raises
        ------
        NotImplementedError
            If offload is requested.
        AssertionError
            If the file is not a safetensors file.
        """
        device = kwargs.get("device", "cpu")
        offload = kwargs.get("offload", False)

        if offload:
            raise NotImplementedError("Offload is not supported for Flux3Transformer2DModel")

        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        assert pretrained_model_name_or_path.is_file() or pretrained_model_name_or_path.name.endswith(
            (".safetensors", ".sft")
        ), "Only safetensors are supported"
        # _build_model already casts to torch_dtype; casting again would repeat diffusers' fp32-module warning
        transformer, model_state_dict, metadata = cls._build_model(pretrained_model_name_or_path, **kwargs)
        quantization_config = json.loads(metadata.get("quantization_config", "{}"))
        rank = quantization_config.get("rank", 32)

        precision = get_precision()
        if precision == "fp4":
            precision = "nvfp4"
        transformer._patch_model(precision=precision, rank=rank)
        transformer = transformer.to_empty(device=device)
        # QK normalization runs in fp32, as in diffusers (``_keep_in_fp32_modules``)
        for name, module in transformer.named_modules():
            if name.rsplit(".", 1)[-1] in cls._keep_in_fp32_modules:
                module.float()

        patch_scale_key(transformer, model_state_dict)
        transformer.load_state_dict(model_state_dict)

        return transformer
