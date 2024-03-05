# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Base classes for various fairseq models.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from search_spaces.hat.fairseq import utils
from search_spaces.hat.fairseq.data import Dictionary
from search_spaces.hat.fairseq.models import FairseqDecoder, FairseqEncoder
from predictors.hat.latency_predictor import Net

import os
from optimizers.optim_factory import get_sampler

class BaseFairseqModel(nn.Module):
    """Base class for fairseq models."""

    def __init__(self):
        super().__init__()
        self._is_generation_fast = False

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        pass

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        raise NotImplementedError('Model must implement the build_model method')

    def get_targets(self, sample, net_output):
        """Get targets from either the sample or the net's output."""
        return sample['target']

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        if hasattr(self, 'decoder'):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        elif torch.is_tensor(net_output):
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def extract_features(self, *args, **kwargs):
        """Similar to *forward* but only return features."""
        return self(*args, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return None

    def load_state_dict(self, state_dict, strict=True):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """
        self.upgrade_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict)

    def upgrade_state_dict(self, state_dict):
        """Upgrade old state dicts to work with newer code."""
        self.upgrade_state_dict_named(state_dict, '')

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade old state dicts to work with newer code.

        Args:
            state_dict (dict): state dictionary to upgrade, in place
            name (str): the state dict key corresponding to the current module
        """
        assert state_dict is not None

        def do_upgrade(m, prefix):
            if len(prefix) > 0:
                prefix += '.'

            for n, c in m.named_children():
                name = prefix + n
                if hasattr(c, 'upgrade_state_dict_named'):
                    c.upgrade_state_dict_named(state_dict, name)
                elif hasattr(c, 'upgrade_state_dict'):
                    c.upgrade_state_dict(state_dict)
                do_upgrade(c, name)

        do_upgrade(self, name)

    def make_generation_fast_(self, **kwargs):
        """Optimize model for faster generation."""
        if self._is_generation_fast:
            return  # only apply once
        self._is_generation_fast = True

        # remove weight norm from all modules in the network
        def apply_remove_weight_norm(module):
            try:
                nn.utils.remove_weight_norm(module)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(apply_remove_weight_norm)

        seen = set()

        def apply_make_generation_fast_(module):
            if module != self and hasattr(module, 'make_generation_fast_') \
                    and module not in seen:
                seen.add(module)
                module.make_generation_fast_(**kwargs)

        self.apply(apply_make_generation_fast_)

        def train(mode=True):
            if mode:
                raise RuntimeError('cannot train after make_generation_fast')

        # this model should no longer be used for training
        self.eval()
        self.train = train

    def prepare_for_onnx_export_(self, **kwargs):
        """Make model exportable via ONNX trace."""
        seen = set()

        def apply_prepare_for_onnx_export_(module):
            if module != self and hasattr(module, 'prepare_for_onnx_export_') \
                    and module not in seen:
                seen.add(module)
                module.prepare_for_onnx_export_(**kwargs)

        self.apply(apply_prepare_for_onnx_export_)

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', **kwargs):
        """
        Load a :class:`~fairseq.models.FairseqModel` from a pre-trained model
        file. Downloads and caches the pre-trained model file if needed.

        The base implementation returns a
        :class:`~fairseq.hub_utils.GeneratorHubInterface`, which can be used to
        generate translations or sample from language models. The underlying
        :class:`~fairseq.models.FairseqModel` can be accessed via the
        *generator.models* attribute.

        Other models may override this to implement custom hub interfaces.

        Args:
            model_name_or_path (str): either the name of a pre-trained model to
                load or a path/URL to a pre-trained model state dict
            checkpoint_file (str, optional): colon-separated list of checkpoint
                files in the model archive to ensemble (default: 'model.pt')
            data_name_or_path (str, optional): point args.data to the archive
                at the given path/URL. Can start with '.' or './' to reuse the
                model archive path.
        """
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            **kwargs,
        )
        print(x['args'])
        return hub_utils.GeneratorHubInterface(x['args'], x['task'], x['models'])

    @classmethod
    def hub_models(cls):
        return {}


class FairseqEncoderDecoderModel(BaseFairseqModel):
    """Base class for encoder-decoder models.

    Args:
        encoder (FairseqEncoder): the encoder
        decoder (FairseqDecoder): the decoder
    """

    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.sampler = get_sampler("reinmax")
        self.predictor = Net(101, 400, 6, hw_embed_on=True, hw_embed_dim=10).cuda()#.half()
        # turn off grad for predictor
        for param in self.predictor.parameters():
            param.requires_grad = False


        assert isinstance(self.encoder, FairseqEncoder)
        assert isinstance(self.decoder, FairseqDecoder)

    def preprocess_for_predictor(self, arch_param, choices):

        arch_param_encoder_dim = arch_param["encoder-embed-dim"]
        arch_param_decoder_dim = arch_param["decoder-embed-dim"]
        arch_param_encoder_layer_num = arch_param["encoder-layer-num"]
        arch_param_decoder_layer_num = arch_param["decoder-layer-num"]
        arch_param_encoder_ffn_embed_dim = torch.zeros_like(arch_param["encoder-ffn-embed-dim"]).cuda()
        layer_num_encoder = choices["encoder-layer-num-choice"][torch.argmax(arch_param_encoder_layer_num).item()]
        for i in range(layer_num_encoder):
             arch_param_encoder_ffn_embed_dim[i]= arch_param["encoder-ffn-embed-dim"][i]
        arch_param_decoder_ffn_embed_dim = torch.zeros_like(arch_param["decoder-ffn-embed-dim"]).cuda()
        layer_num_decoder = choices["decoder-layer-num-choice"][torch.argmax(arch_param_decoder_layer_num).item()]
        for i in range(layer_num_decoder):
                arch_param_decoder_ffn_embed_dim[i]= arch_param["decoder-ffn-embed-dim"][i]
        arch_param_encoder_self_attention_heads = torch.zeros_like(arch_param["encoder-self-attention-heads"]).cuda()
        layer_num_encoder = choices["encoder-layer-num-choice"][torch.argmax(arch_param_encoder_layer_num).item()]
        for i in range(layer_num_encoder):
                arch_param_encoder_self_attention_heads[i]= arch_param["encoder-self-attention-heads"][i]
        arch_param_decoder_self_attention_heads = torch.zeros_like(arch_param["decoder-self-attention-heads"]).cuda()
        layer_num_decoder = choices["decoder-layer-num-choice"][torch.argmax(arch_param_decoder_layer_num).item()]
        for i in range(layer_num_decoder):
                arch_param_decoder_self_attention_heads[i]= arch_param["decoder-self-attention-heads"][i]
        arch_param_decoder_ende_attention_heads = torch.zeros_like(arch_param["decoder-ende-attention-heads"]).cuda()
        layer_num_decoder = choices["decoder-layer-num-choice"][torch.argmax(arch_param_decoder_layer_num).item()]
        for i in range(layer_num_decoder):
                arch_param_decoder_ende_attention_heads[i]= arch_param["decoder-ende-attention-heads"][i]
        arch_param_decoder_arbitrary_ende_attn = torch.zeros_like(arch_param["decoder-arbitrary-ende-attn"]).cuda()
        layer_num_decoder = choices["decoder-layer-num-choice"][torch.argmax(arch_param_decoder_layer_num).item()]
        for i in range(layer_num_decoder):
                arch_param_decoder_arbitrary_ende_attn[i]= arch_param["decoder-arbitrary-ende-attn"][i]
        


        #arch_param_decoder_ffn_embed_dim = arch_param["decoder-ffn-embed-dim"]
        #arch_param_encoder_self_attention_heads = arch_param["encoder-self-attention-heads"]
        #arch_param_decoder_self_attention_heads = arch_param["decoder-self-attention-heads"]
        #arch_param_decoder_ende_attention_heads = arch_param["decoder-ende-attention-heads"]
        #arch_param_decoder_arbitrary_ende_attn = arch_param["decoder-arbitrary-ende-attn"]
        # concatenate all in flat list
        arch_param_list = []
        arch_param_list.append(arch_param_encoder_dim)
        arch_param_list.append(arch_param_decoder_dim)
        arch_param_list.append(arch_param_encoder_layer_num)
        arch_param_list.append(arch_param_decoder_layer_num)
        arch_param_list.append(arch_param_encoder_ffn_embed_dim.flatten())
        arch_param_list.append(arch_param_decoder_ffn_embed_dim.flatten())
        arch_param_list.append(arch_param_encoder_self_attention_heads.flatten())
        arch_param_list.append(arch_param_decoder_self_attention_heads.flatten())
        arch_param_list.append(arch_param_decoder_ende_attention_heads.flatten())
        arch_param_list.append(arch_param_decoder_arbitrary_ende_attn.flatten())
        # concat all in one tensor
        arch_param_tensor = torch.cat(arch_param_list, dim=-1).to(arch_param_encoder_dim.device)
        #print(arch_param_tensor)
        return arch_param_tensor


    def forward(self, src_tokens, src_lengths, prev_output_tokens, arch_param, hw_embed=None, **kwargs):
        """
        Run the forward pass for an encoder-decoder model.

        First feed a batch of source tokens through the encoder. Then, feed the
        encoder output and previous decoder outputs (i.e., teacher forcing) to
        the decoder to produce the next outputs::

            encoder_out = self.encoder(src_tokens, src_lengths)
            return self.decoder(prev_output_tokens, encoder_out)

        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        #hw_embed = torch.randn(1, 10)
        if self.training:
            arch_params_sampled = {}
            for name, param in arch_param.items():
                #print(name)
                #print(param)
                arch_params_sampled[name] = self.sampler.sample(param)#.half()
        else:
            #descritize the arch_param
            arch_params_sampled = {}
            for name, param in arch_param.items():
                param_to_argmax = torch.zeros_like(param)
                selected_id = torch.argmax(param,dim=-1)
                param_to_argmax.scatter_(dim=-1, index=selected_id.unsqueeze(-1), value=1)
                arch_params_sampled[name] = param_to_argmax#.half()
        #print(arch_params_sampled)
        choices = utils.get_space("space0")
        self.set_sample_config(choices, arch_params_sampled)
        predictor_inp = self.preprocess_for_predictor(arch_params_sampled, choices)
        latency = self.predictor(predictor_inp.unsqueeze(0), hw_embed)
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, arch_param=arch_params_sampled, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, arch_param=arch_params_sampled, **kwargs)
        return decoder_out , latency

    def extract_features(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        features = self.decoder.extract_features(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return features

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return (self.encoder.max_positions(), self.decoder.max_positions())

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()


class SuperFairseqEncoderDecoderModel(FairseqEncoderDecoderModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    def set_sample_config(self, choices:dict, arch_param:dict):
        #print(self.encoder)
        #(self.decoder)
        self.encoder.set_sample_config(choices, arch_param)
        self.decoder.set_sample_config(choices, arch_param)


class FairseqModel(FairseqEncoderDecoderModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        utils.deprecation_warning(
            'FairseqModel is deprecated, please use FairseqEncoderDecoderModel '
            'or BaseFairseqModel instead',
            stacklevel=4,
        )


class FairseqMultiModel(BaseFairseqModel):
    """Base class for combining multiple encoder-decoder models."""

    def __init__(self, encoders, decoders):
        super().__init__()
        assert encoders.keys() == decoders.keys()
        self.keys = list(encoders.keys())
        for key in self.keys:
            assert isinstance(encoders[key], FairseqEncoder)
            assert isinstance(decoders[key], FairseqDecoder)

        self.models = nn.ModuleDict({
            key: FairseqModel(encoders[key], decoders[key])
            for key in self.keys
        })

    @staticmethod
    def build_shared_embeddings(
        dicts: Dict[str, Dictionary],
        langs: List[str],
        embed_dim: int,
        build_embedding: callable,
        pretrained_embed_path: Optional[str] = None,
    ):
        """
        Helper function to build shared embeddings for a set of languages after
        checking that all dicts corresponding to those languages are equivalent.

        Args:
            dicts: Dict of lang_id to its corresponding Dictionary
            langs: languages that we want to share embeddings for
            embed_dim: embedding dimension
            build_embedding: callable function to actually build the embedding
            pretrained_embed_path: Optional path to load pretrained embeddings
        """
        shared_dict = dicts[langs[0]]
        if any(dicts[lang] != shared_dict for lang in langs):
            raise ValueError(
                '--share-*-embeddings requires a joined dictionary: '
                '--share-encoder-embeddings requires a joined source '
                'dictionary, --share-decoder-embeddings requires a joined '
                'target dictionary, and --share-all-embeddings requires a '
                'joint source + target dictionary.'
            )
        return build_embedding(
            shared_dict, embed_dim, pretrained_embed_path
        )

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        decoder_outs = {}
        for key in self.keys:
            encoder_out = self.models[key].encoder(src_tokens, src_lengths, **kwargs)
            decoder_outs[key] = self.models[key].decoder(
                prev_output_tokens, encoder_out, **kwargs,
            )
        return decoder_outs

    def max_positions(self):
        """Maximum length supported by the model."""
        return {
            key: (self.models[key].encoder.max_positions(), self.models[key].decoder.max_positions())
            for key in self.keys
        }

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return min(model.decoder.max_positions() for model in self.models.values())

    @property
    def encoder(self):
        return self.models[self.keys[0]].encoder

    @property
    def decoder(self):
        return self.models[self.keys[0]].decoder


class FairseqLanguageModel(BaseFairseqModel):
    """Base class for decoder-only models.

    Args:
        decoder (FairseqDecoder): the decoder
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        assert isinstance(self.decoder, FairseqDecoder)

    def forward(self, src_tokens, **kwargs):
        """
        Run the forward pass for a decoder-only model.

        Feeds a batch of tokens through the decoder to predict the next tokens.

        Args:
            src_tokens (LongTensor): tokens on which to condition the decoder,
                of shape `(batch, tgt_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`

        Returns:
            tuple:
                - the decoder's output of shape `(batch, seq_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        return self.decoder(src_tokens, **kwargs)

    def extract_features(self, src_tokens, **kwargs):
        """
        Similar to *forward* but only return features.

        Returns:
            tuple:
                - the decoder's features of shape `(batch, seq_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        return self.decoder.extract_features(src_tokens, **kwargs)

    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.decoder.max_positions()

    def max_decoder_positions(self):
        """Maximum length supported by the decoder."""
        return self.decoder.max_positions()

    @property
    def supported_targets(self):
        return {'future'}


class FairseqEncoderModel(BaseFairseqModel):
    """Base class for encoder-only models.

    Args:
        encoder (FairseqEncoder): the encoder
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        assert isinstance(self.encoder, FairseqEncoder)

    def forward(self, src_tokens, src_lengths, **kwargs):
        """
        Run the forward pass for a encoder-only model.

        Feeds a batch of tokens through the encoder to generate features.

        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            src_lengths (LongTensor): source sentence lengths of shape `(batch)`

        Returns:
            the encoder's output, typically of shape `(batch, src_len, features)`
        """
        return self.encoder(src_tokens, src_lengths, **kwargs)

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        encoder_out = net_output['encoder_out']
        if torch.is_tensor(encoder_out):
            logits = encoder_out.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def max_positions(self):
        """Maximum length supported by the model."""
        return self.encoder.max_positions()
