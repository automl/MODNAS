# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

"""Full definition of a decoder-only transformer-based language model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from typing_extensions import Self
import pickle
from search_spaces.gpt.config import Config
from search_spaces.gpt.blocks import *
from search_spaces.gpt.super_modules.embedding_super import SuperEmbedding
from search_spaces.gpt.super_modules.rotary_embedding import SuperRotaryEmbedding
from search_spaces.gpt.super_modules.lmhead_super import LMHeadSuper
from predictors.gpt.net import Net
from optimizers.optim_factory import get_sampler
class GPT(nn.Module):
    def __init__(self, config: Config, choices_dict: dict, ignore_index: bool) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config
        self.sampler = get_sampler("reinmax")
        self.lm_head = LMHeadSuper(config.n_embd, config.padded_vocab_size, config.lm_head_bias)
        self.rotary_embeddings = [SuperRotaryEmbedding(config, config.block_size) for _ in range(config.n_layer)]
        self.rotary_dummy = SuperRotaryEmbedding(config,config.block_size )
        self.transformer = nn.ModuleDict(
            dict(
                wte=SuperEmbedding(config.padded_vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config, self.rotary_embeddings[i]) for i in range(config.n_layer)),
                ln_f=self.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.loss_traj = []
        self.obj = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean', label_smoothing=0.0)
        self.max_layer = config.n_layer
        self.choices = choices_dict
        self.max_seq_length = self.config.block_size
        self.mask_cache: Optional[torch.Tensor] = None
        
        self.sample_embed_dim = None  # type: Optional[int]
        self.sample_intermediate_size = None
        self.sample_num_heads = None
        self.transformer.wte.weight = self.lm_head.weight
        self.predictor = Net(80, 256, 4, True, 10)
        self.predictor = self.predictor.to(config.device)
        self.predictor.load_state_dict(torch.load('predictors/gpt/metapredictor.pt'))
        for param in self.predictor.parameters():
            param.requires_grad = False

    def normalize_energy(self, energy_pred, device_name):
        with open("max_min_energy_stats.pkl","rb") as f:
            max_min_energy_stats = pickle.load(f)
        max_energy = max_min_energy_stats["max"][device_name]
        min_energy = max_min_energy_stats["min"][device_name]
        energy_pred = (energy_pred - min_energy)/(max_energy - min_energy)
        return energy_pred

    def get_and_set_sampled_arch_config(self, arch_params):
        layer_param, head_param, mlp_ratio_param, embed_dim_param, bias_param = arch_params

        argmax_layer = torch.argmax(layer_param, dim=-1)
        layer_selected = layer_param[argmax_layer]
        layer_num = self.choices['n_layer_choices'][argmax_layer]
        argmax_embed_dim = torch.argmax(embed_dim_param, dim=-1)
        embed_dim_selected = embed_dim_param[argmax_embed_dim]
        #print(argmax_embed_dim)
        embed_dim_selected = self.choices['embed_dim_choices'][argmax_embed_dim]
        argmax_bias = torch.argmax(bias_param, dim=-1)
        bias_selected = bias_param[argmax_bias]
        bias_selected = self.choices['bias_choices'][argmax_bias]
        argmax_head = torch.argmax(head_param, dim=-1)
        heads_selected = head_param[:,argmax_head]
        heads_selected = [self.choices['n_head_choices'][i] for i in argmax_head]
        argmax_mlp_ratio = torch.argmax(mlp_ratio_param, dim=-1)
        mlp_ratio_selected = mlp_ratio_param[:,argmax_mlp_ratio]
        mlp_ratio_selected = [self.choices['mlp_ratio_choices'][i] for i in argmax_mlp_ratio]
        intermediate_size = [embed_dim_selected * mlp_ratio for mlp_ratio in mlp_ratio_selected]
        print(embed_dim_selected, intermediate_size, heads_selected, layer_num, bias_selected, list(range(layer_num)))
        self.set_sample_config(embed_dim_selected, intermediate_size, heads_selected, layer_num, bias_selected, list(range(layer_num)))

    def get_arch_config(self, arch_params):
        layer_param, head_param, mlp_ratio_param, embed_dim_param, bias_param = arch_params

        argmax_layer = torch.argmax(layer_param, dim=-1)
        layer_selected = layer_param[argmax_layer]
        layer_num = self.choices['n_layer_choices'][argmax_layer]
        argmax_embed_dim = torch.argmax(embed_dim_param, dim=-1)
        embed_dim_selected = embed_dim_param[argmax_embed_dim]
        #print(argmax_embed_dim)
        embed_dim_selected = self.choices['embed_dim_choices'][argmax_embed_dim]
        argmax_bias = torch.argmax(bias_param, dim=-1)
        bias_selected = bias_param[argmax_bias]
        bias_selected = self.choices['bias_choices'][argmax_bias]
        argmax_head = torch.argmax(head_param, dim=-1)
        heads_selected = head_param[:,argmax_head]
        heads_selected = [self.choices['n_head_choices'][i] for i in argmax_head]
        argmax_mlp_ratio = torch.argmax(mlp_ratio_param, dim=-1)
        mlp_ratio_selected = mlp_ratio_param[:,argmax_mlp_ratio]
        mlp_ratio_selected = [self.choices['mlp_ratio_choices'][i] for i in argmax_mlp_ratio]
        intermediate_size = [embed_dim_selected * mlp_ratio for mlp_ratio in mlp_ratio_selected]

        #sampled_arch = {"sample_embed_dim": embed_dim_selected, "sample_n_layer": layer_num, "sample_n_head": heads_selected, 'sample_mlp_ratio':mlp_ratio_selected, "sample_bias":bias_selected}
        return sampled_arch

    def get_arch_one_hot(self, arch_params):
        layer_param, head_param, mlp_ratio_param, embed_dim_param, bias_param = arch_params
        layer_selected = torch.argmax(layer_param, dim=-1)
        layer_num = self.choices['n_layer_choices'][layer_selected]
        mlp_ratio_new = torch.zeros_like(mlp_ratio_param)
        num_heads_new = torch.zeros_like(head_param)
        for i in range(layer_num):
            mlp_ratio_new[i] = mlp_ratio_param[i]
            num_heads_new[i] = head_param[i]
        one_hot = torch.cat([embed_dim_param, layer_param, mlp_ratio_new.view(-1), num_heads_new.view(-1), bias_param])
        return one_hot

    @property
    def norm_class(self):
        # `self._norm_class` cannot be the type to keep the config json serializable
        from search_spaces.gpt.super_modules.rmsnorm_super import RMSNormSuper
        from search_spaces.gpt.super_modules.layernorm_super import LayerNormSuper
        if self.config._norm_class == "RMSNorm":

            return RMSNormSuper
        return LayerNormSuper

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length
    
    def set_sample_config(self, sample_embed_dim: int, sample_intermediate_size: list, sample_num_heads: list, sample_n_layer: int, sample_bias_flag: bool, sample_layer_indices: list) -> None:
        self.sample_embed_dim = sample_embed_dim
        self.sample_intermediate_size = sample_intermediate_size
        self.sample_num_heads = sample_num_heads
        self.sample_n_layer = sample_n_layer
        self.sample_bias_flag = sample_bias_flag
        self.sample_layer_indices = sample_layer_indices
        self.transformer.wte.set_sample_config(sample_embed_dim)
        self.transformer.ln_f.set_sample_config(sample_embed_dim)
        self.rotary_dummy.set_sample_config(self.config.n_embd, self.config.n_head)
        #self.rotary_embedding.set_sample_config(self.config.n_embd, self.config.n_head)
        #print(sample_layer_indices)
        for i in sample_layer_indices:
            block = self.transformer.h[i]
            block.set_sample_config(sample_embed_dim, sample_intermediate_size[i], sample_num_heads[i], sample_bias_flag)
        self.lm_head.set_sample_config(sample_embed_dim, sample_bias_flag)

    @max_seq_length.setter
    def max_seq_length(self, value: int) -> None:
        """
        When doing inference, the sequences used might be shorter than the model's context length.
        This allows setting a smaller number to avoid allocating unused memory
        """
        if value > self.config.block_size:
            raise ValueError(f"Cannot attend to {value}, block size is only {self.config.block_size}")
        self._max_seq_length = value
        if not hasattr(self, "cos"):
            # first call
            self.rotary_dummy.set_sample_config(self.config.n_embd, self.config.n_head)
            cos, sin = self.rotary_dummy.rope_cache()
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)
        # override
        elif value != self.cos.size(0):
            self.cos, self.sin = self.rotary_dummy.rope_cache(device=self.cos.device)
        # the mask and kv cache size will get updated on `set_kv_cache`. we cannot update it here because we don't know
        # if the kv cache is expected


    def _init_weights(self, module: nn.Module) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, arch_params, hw_embed, device_name, labels, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        T = idx.size(1)
        layer_param, head_param, mlp_ratio_param, embed_dim_param, bias_param = arch_params
        layer_param, head_param, mlp_ratio_param, embed_dim_param, bias_param = self.sampler.sample_step([layer_param, head_param, mlp_ratio_param, embed_dim_param, bias_param])
        self.get_and_set_sampled_arch_config([layer_param, head_param, mlp_ratio_param, embed_dim_param, bias_param])
        arch_one_hot = self.get_arch_one_hot([layer_param, head_param, mlp_ratio_param, embed_dim_param, bias_param])
        energy_pred = self.predictor(arch_one_hot.unsqueeze(0), hw_embed)
        
        argmax_layer = torch.argmax(layer_param, dim=-1)
        layer_selected = layer_param[argmax_layer]
        argmax_embed_dim = torch.argmax(embed_dim_param, dim=-1)
        embed_dim_selected = embed_dim_param[argmax_embed_dim]
        argmax_bias = torch.argmax(bias_param, dim=-1)
        bias_selected = bias_param[argmax_bias]
        head_selected = []
    
        for i in range(self.max_layer):
            argmax_head = torch.argmax(head_param[i], dim=-1)
            head_selected.append(head_param[i][argmax_head])
        mlp_ratio_selected = []
        for i in range(self.max_layer):
            argmax_mlp_ratio = torch.argmax(mlp_ratio_param[i], dim=-1)
            mlp_ratio_selected.append(mlp_ratio_param[i][argmax_mlp_ratio])
        #print(layer_selected, embed_dim_selected, bias_selected, head_selected, mlp_ratio_selected)
        if self.max_seq_length < T:
            raise ValueError(f"Cannot forward sequence of length {T}, max seq length is only {self.max_seq_length}.")
        #self.reset_parameters() #TODO: we need to reset rope cache every time (might be inefficient)
        if input_pos is not None:  # use the kv cache
            #cos = self.cos.index_select(0, input_pos)
            #sin = self.sin.index_select(0, input_pos)
            if self.mask_cache is None:
                raise TypeError("You need to call `gpt.set_kv_cache()`")
            mask = self.mask_cache.index_select(2, input_pos)
        else:
            #cos = self.cos[:T]
            #sin = self.sin[:T]
            mask = None

        x = self.transformer.wte(idx) * embed_dim_selected # token embeddings of shape (b, t, n_embd)
        #print(self.sample_layer_indices)
        if self.config.scale_embeddings:
            x = x * (self.config.n_embd**0.5)
        for i in self.sample_layer_indices:
            block = self.transformer.h[i]
            #print(head_selected[i], mlp_ratio_selected[i], bias_selected)
            x = block(x,  mask, input_pos, embed_dim_selected, head_selected[i], mlp_ratio_selected[i], bias_selected)
        x = x * layer_selected
        x = self.transformer.ln_f(x) * embed_dim_selected
        logits = self.lm_head(x) * embed_dim_selected
        loss = self.obj(logits.view(-1, logits.size(-1)), labels)
        # normalize the loss
        if len(self.loss_traj) > 10:
            energy_pred = self.normalize_energy(energy_pred, device_name)
            loss_min = min(self.loss_traj)
            loss_max = max(self.loss_traj)
            energy_pred = energy_pred*(loss_max-loss_min)+loss_min
        if self.training:
            self.loss_traj.append(loss.detach().item())
        if len(self.loss_traj) > 100:
            self.reset_loss_traj()
        return loss, energy_pred, logits
    
    def reset_loss_traj(self):
        self.loss_traj = []
    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def set_kv_cache(
        self,
        batch_size: int,
        rope_cache_length: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        '''if rope_cache_length is None:
            rope_cache_length = self.cos.size(-1)
        max_seq_length = self.max_seq_length

        # initialize the kv cache for all blocks
        for block in self.transformer.h:
            block.attn.kv_cache = block.attn.build_kv_cache(
                batch_size, max_seq_length, rope_cache_length, device, dtype
            )'''

        if self.mask_cache is None or self.mask_cache.size(3) != max_seq_length:
            # passing `attn_mask` to SDPA disables the flash implementation. since we only need the mask
            # for the kv-cache support (only during inference), we only create it in that situation
            self.mask_cache = build_mask_cache(self.max_seq_length, device)

    def clear_kv_cache(self) -> None:
        self.mask_cache = None
        #for block in self.transformer.h:
        #    block.attn.kv_cache = None



def build_mask_cache(max_seq_length: int, device: Optional[torch.device] = None) -> torch.Tensor:
    ones = torch.ones((max_seq_length, max_seq_length), device=device, dtype=torch.bool)
    return torch.tril(ones).unsqueeze(0).unsqueeze(0)

