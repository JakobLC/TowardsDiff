from abc import abstractmethod

import sys
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from source.utils.argparsing import TieredParser
from source.utils.fp16 import convert_module_to_f16, convert_module_to_f32
from source.utils.mixed import model_arg_is_trivial, nice_split, assert_one_to_one_list_of_str
from source.models.nn import (SiLU,conv_nd,linear,avg_pool_nd,zero_module,normalization,
                              timestep_embedding)
import pandas as pd
import warnings
from argparse import Namespace

tp = TieredParser()

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, x_attn=None):
        for layer in self:
            if isinstance(layer, AttentionBlock):
                if x_attn is None:
                    x = layer(x)
                else:
                    x = layer(x,y=x_attn)
            elif isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, channels_out=None, dims=2, mode="nearest"):
        super().__init__()
        if channels_out is None:
            channels_out = channels
        self.mode = mode
        self.channels_out = channels_out
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels_out, channels_out, 3, padding=1)
        if channels_out != channels:
            self.channel_mapper = conv_nd(dims, channels, channels_out, 1)
    def forward(self, x):
        assert x.shape[1] == self.channels

        if hasattr(self, "channel_mapper"):
            x = self.channel_mapper(x)
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode=self.mode
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.mode)
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, channels_out=None, dims=2):
        super().__init__()
        if channels_out is None:
            channels_out = channels
        self.channels_out = channels_out
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)
        if channels_out != channels:
            self.channel_mapper = conv_nd(dims, channels, channels_out, 1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        x = self.op(x)
        if hasattr(self, "channel_mapper"):
            x = self.channel_mapper(x)
        return x

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return self._forward(x, emb)

    def _forward(self, x, emb):
        h = self.in_layers(x)
        
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class MLPBlock(TimestepBlock):
    """
    Based on the MLP block from SiD (simple diffusion) pseudo code.

    def mlp_block(x, emb, expansion_factor=4):
    B, HW, C = x.shape
    x = Normalize(x)
    mlp_h = Dense(x, expansion_factor * C)
    scale = DenseGeneral(emb, mlp_h.shape [2:])
    shift = DenseGeneral(emb, mlp_h.shape [2:])
    mlp_h = swish(mlp_h)
    mlp_h = mlp_h * (1. + scale [:, None ]) + shift [:, None]
    if config.transformer_dropout > 0.:
        mlp_h = Dropout(mlp_h, config.transformer_dropout)
    out = Dense(mlp_h, C, kernel_init = zeros)
    return out"""
    def __init__(
        self,
        channels,
        emb_channels,
        expansion_factor=4,
        dropout=0.0,
        out_channels=None,
        use_scale_shift_norm=False,
        dims=2,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        c = expansion_factor * channels
        self.in_layers = nn.Sequential(
            normalization(channels),
            conv_nd(dims, channels, c, 1),
            SiLU(),
        )
        self.emb_layers = linear(
                emb_channels,
                2 * c if use_scale_shift_norm else c,
            )
        self.out_layers = nn.Sequential(
            nn.Dropout(p=dropout),
            conv_nd(dims, c, self.out_channels, 1),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
        
    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return self._forward(x, emb)

    
    def _forward(self, x, emb):
        #b, c, *spatial = x.shape
        #x = x.reshape(b, c, -1)
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = h * (1 + scale) + shift
        else:
            h = h + emb_out
        h = self.out_layers(h)
        return (self.skip_connection(x) + h)#.reshape(b, c, *spatial)

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, with_xattn=False, xattn_channels=None):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.with_xattn = with_xattn
        if self.with_xattn:
            if xattn_channels is None:
                xattn_channels = channels
            self.xattn_channels = xattn_channels
            self.qk_x = conv_nd(1, xattn_channels, 2*channels, 1) 
            self.v_x = conv_nd(1, channels, channels, 1)
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return self._forward(x)

    def _forward(self, x, y=None):
        b, c, *spatial = x.shape
        #assert c==self.channels, f"expected {self.channels} channels, got {c} channels"
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        if y is not None:
            assert self.with_xattn, "y is is only supported as an input for AttentionBlocks with cross attention"
            b, cx, *spatial2 = y.shape
            assert cx==self.xattn_channels, f"expected {self.xattn_channels} channels, got {cx} channels"
            y = y.reshape(b, cx, -1)
            qk = self.qk_x(self.norm(y))
            v = self.v_x(self.norm(h))
            qkv_x = torch.cat([qk,v],dim=-1).reshape(b * self.num_heads, -1, qk.shape[2])
            h = self.attention(qkv_x)+h
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = torch.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        return torch.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += torch.DoubleTensor([matmul_ops])
            
class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention__init__
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        image_size=32,
        out_channels=1,
        model_channels=64,
        num_res_blocks=3,
        num_middle_res_blocks=2,
        attention_resolutions="-2,-1",
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        final_act="none",
        unet_input_dict=None,
    ):
        super().__init__()
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        self.image_size = image_size
        time_embed_dim = model_channels*4
        assert isinstance(unet_input_dict,dict), "unet_input_dict must be a dictionary"
        assert len(unet_input_dict)>0, "unet_input_dict must have at least one entry"
        self.legal_keys = list(unet_input_dict.keys())
        if isinstance(num_res_blocks,int):
            num_res_blocks = [num_res_blocks]*len(channel_mult)
        assert len(num_res_blocks) == len(channel_mult), f"len(num_res_blocks): {len(num_res_blocks)} must be equal to len(channel_mult): {len(channel_mult)}"

        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks

        self.attention_resolutions = []
        if attention_resolutions not in ["", ","]:
            for ar in attention_resolutions.split(","):
                ar = int(ar)
                if ar < 0:
                    ar = len(channel_mult) + ar
                self.attention_resolutions.append(ar)
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.unet_input_dict = unet_input_dict

        self.class_dict = {}
        self.use_class_emb = False

        for k,v in self.unet_input_dict.items():
            if v["input_type"] == "scalar_discrete":
                assert v["size"]>0, f"class size must be greater than 0. got {v['size']} for class {k}"
                self.class_dict[k] = v["size"]
                self.use_class_emb = True

        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        self.fp16_attrs = ["input_blocks","output_blocks"]
        if num_middle_res_blocks>=1:
            self.fp16_attrs.append("middle_block")

        self.in_channels = 0
        for k,v in self.unet_input_dict.items():
            c = v.get("in_chans",0)
            imsize = v.get("img_size",self.image_size)
            if c>0:
                self.unet_input_dict[k]["slice"] = slice(self.in_channels,self.in_channels+c)
                self.in_channels += c
            assert imsize==self.image_size, "all image sizes must be the same. kwarg "+k+" has image size "+str(imsize)+" while the model has image size "+str(self.image_size)
        self.fp16_attrs.append("time_embed")
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        
        if self.use_class_emb:
            self.fp16_attrs.append("class_emb")
            self.class_emb = nn.ModuleDict()
            for k,nc in self.class_dict.items():
                self.class_emb[k] = nn.Embedding(nc+1, time_embed_dim)
        
                
        embed_out = nn.Identity()
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, self.in_channels, model_channels, 3, padding=1))
            ])
        self.input_skip = [False]
        input_block_chans = [model_channels]
        ch = model_channels
        res_block_kwargs = {"emb_channels": time_embed_dim,
                            "dropout": dropout,
                            "dims": dims,
                            "use_scale_shift_norm": use_scale_shift_norm}
        attn_kwargs = {"num_heads": num_heads,
                        "with_xattn": False,
                        "xattn_channels": None}
        resolution = 0
        
        assert channel_mult[0]==1, "channel_mult[0] must be 1"
        for level, (mult, n_res_blocks) in enumerate(zip(channel_mult, num_res_blocks)):
            for _ in range(n_res_blocks):
                ch_in = ch
                ch = mult*model_channels
                layers = []
                
                if resolution in self.attention_resolutions:
                    layers = [ResBlock(ch_in,out_channels=ch,**res_block_kwargs),
                              AttentionBlock(ch,**attn_kwargs)]
                else:
                    layers = [ResBlock(ch_in,out_channels=ch,**res_block_kwargs)]
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.input_skip.append(False)
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                resolution += 1
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims, channels_out=None))
                )
                self.input_skip[-1] = True
                self.input_skip.append(False)
                input_block_chans.append(ch)
        if resolution in self.attention_resolutions:
            middle_layers = (sum([[ResBlock(ch,**res_block_kwargs),
                           AttentionBlock(ch,**attn_kwargs)] 
                           for _ in range(num_middle_res_blocks-1)],[])+
                        [ResBlock(ch,**res_block_kwargs)])
        else:
            middle_layers = [ResBlock(ch,**res_block_kwargs) for _ in range(num_middle_res_blocks)]
        """for i in range(num_middle_res_blocks):
            middle_layers.append(ResBlock(ch,**res_block_kwargs))
            if len(self.attention_resolutions)>0 and i < num_middle_res_blocks-1:
                middle_layers.append(AttentionBlock(ch,**attn_kwargs))"""
        self.middle_block = TimestepEmbedSequential(*middle_layers)

        attn_kwargs["num_heads"] = num_heads_upsample
        self.output_blocks = nn.ModuleList([])
        for level, mult, n_res_blocks in zip(reversed(list(range(len(channel_mult)))),channel_mult[::-1],num_res_blocks[::-1]):
            for i in range(n_res_blocks + 1):
                ch_in = ch+input_block_chans.pop()
                ch = model_channels * mult
                if resolution in self.attention_resolutions:
                    layers = [ResBlock(ch_in,out_channels=ch,**res_block_kwargs),
                              AttentionBlock(ch,**attn_kwargs)]
                else:
                    layers = [ResBlock(ch_in,out_channels=ch,**res_block_kwargs)]
                if level and i == n_res_blocks:
                    resolution -= 1
                    layers.append(Upsample(ch, conv_resample, dims=dims, channels_out=None,
                                           mode="nearest"))
                    
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.input_skip = [True for _ in self.input_skip]

        final_act_dict = {"none": nn.Identity(),
                        "softmax": nn.Softmax(dim=1),
                        "tanh": nn.Tanh(),
                        "sigmoid": nn.Sigmoid()}
        self.out = nn.Sequential(
            embed_out,
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)),
            final_act_dict[final_act.lower()]
        )
        self.out_channels = out_channels
        self.make_block_info()

    def make_block_info(self,as_df=True):
        block_info_keys = ["depth","resolution",
                           "subclasses","image_size_in","image_size_out",
                           "ch_in","ch_out","backbone_part","has_attention"]
        self.block_info = {k: [] for k in block_info_keys}

        depth = 0
        resolution = 0

        subclasses = []
        image_size_in = self.image_size
        image_size_out = self.image_size

        ch_in = self.model_channels
        ch_out = self.model_channels
        backbone_part = "input"
        has_attention = False
        

        for i in range(len(self.input_blocks)): 
            subclasses = [type(m).__name__ for m in self.input_blocks[i]]
            has_attention = "AttentionBlock" in subclasses
            has_downsample = "Downsample" in subclasses
            if has_downsample:
                resolution += 1
                image_size_out //= 2
            ch_in, ch_out = get_ch_in_out(self.input_blocks[i])
            values = [depth,resolution,subclasses,image_size_in,image_size_out,ch_in,ch_out,backbone_part,has_attention]
            self.block_info = {k: v+[val] for k,v,val in zip(block_info_keys,self.block_info.values(),values)}
            depth += 1
            if has_downsample:
                image_size_in //= 2
        backbone_part = "middle"
        has_downsample = False
        subclasses = [type(m).__name__ for m in self.middle_block]
        has_attention = "AttentionBlock" in subclasses
        ch_in, ch_out = get_ch_in_out(self.middle_block)
        values = [depth,resolution,subclasses,image_size_in,image_size_out,ch_in,ch_out,backbone_part,has_attention]
        self.block_info = {k: v+[val] for k,v,val in zip(block_info_keys,self.block_info.values(),values)}
        depth += 1
        backbone_part = "output"
        for i in range(len(self.output_blocks)):
            subclasses = [type(m).__name__ for m in self.output_blocks[i]]
            has_attention = "AttentionBlock" in subclasses
            has_upsample = "Upsample" in [type(m).__name__ for m in self.output_blocks[i]]
            if has_upsample:
                resolution -= 1
                image_size_out *= 2
            ch_in, ch_out = get_ch_in_out(self.output_blocks[i])
            values = [depth,resolution,subclasses,image_size_in,image_size_out,ch_in,ch_out,backbone_part,has_attention]
            self.block_info = {k: v+[val] for k,v,val in zip(block_info_keys,self.block_info.values(),values)}
            depth += 1
            if has_upsample:
                image_size_in *= 2
        backbone_part = "out"
        subclasses = [type(m).__name__ for m in self.out]
        has_attention = False
        ch_in = ch_out
        ch_out = self.out_channels
        values = [depth,resolution,subclasses,image_size_in,image_size_out,ch_in,ch_out,backbone_part,has_attention]
        self.block_info = {k: v+[val] for k,v,val in zip(block_info_keys,self.block_info.values(),values)}


        if as_df:
            self.block_info = pd.DataFrame(self.block_info)
        #print as df
        #print(self.block_info)
        #assert 0

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        for attr in self.fp16_attrs:
            getattr(self,attr).apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        for attr in self.fp16_attrs:
            getattr(self,attr).apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def apply_class_emb(self, classes):
        emb = 0
        for i,k in enumerate(self.class_dict.keys()):
            emb += self.class_emb[k](classes[:,i])
        return emb

    def forward(self, sample, timesteps, **kwargs):
        """
        Apply the model to an input batch.

        :param sample: an [N x C x ...] Diffusion sample tensor.
        :param timesteps: a 1-D batch of timesteps.
        :param kwargs: additional kwargs for the model. see self.unet_input_dict for available kwargs.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if not all([(k in self.legal_keys) or (kwargs[k] is None) for k in kwargs.keys()]):
            illegal_args = [k for k in kwargs.keys() if (k not in self.legal_keys) and (kwargs[k] is not None)]
            raise ValueError(f"illegal kwargs (= different from None and without model support): {illegal_args}. legal kwargs: {self.legal_keys}")

        h,timesteps,classes = self.prepare_inputs(sample, timesteps, **kwargs)
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.use_class_emb:
            emb = emb + self.apply_class_emb(classes)
        hs = []
        depth = 0
        for module,skip in zip(self.input_blocks,self.input_skip):
            h = module(h, emb)
            if skip:
                hs.append(h)
            else:
                hs.append(0)
            depth += 1
        h = self.middle_block(h, emb)
        depth += 1
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            depth += 1
        h = h.type(sample.dtype)
        h = self.out(h)
        return h
    
    def prepare_inputs(self, sample, timesteps, **kwargs):
        """
        prepare inputs for the model.
        Each keyword argument must be either
            - a tensor of shape [batch_size, num_channels, image_size, image_size]
            - None
            - a list where each element is either [num_channels, 
              image_size, image_size] or None
        The function will construct the by formatting the tensors into a tensor of 
        shape [batch_size, sum(num_channels_i), image_size, image_size].
        """
        kwargs["sample"] = sample
        bs = sample.shape[0]
        h = torch.zeros([bs,self.in_channels,self.image_size,self.image_size],device=sample.device).type(self.inner_dtype)
        if self.use_class_emb:
            classes = torch.zeros((bs,len(self.class_dict)),dtype=torch.long,device=sample.device)
        else:
            classes = None
        for k,v in kwargs.items():
            if model_arg_is_trivial(v):
                assert v is None, f"Trivial inputs should be None, found: {v}"
            elif k not in self.unet_input_dict.keys():
                warnings.warn(f"input {k} is not used by the model")
            elif k in self.class_dict.keys():
                if isinstance(v,list):
                    v = torch.tensor([vi if vi is not None else 0 for vi in v],dtype=torch.long,device=sample.device).flatten()
                elif isinstance(v,int):
                    v = v*torch.ones(bs,dtype=torch.long,device=sample.device)
                else:
                    if v.numel() == 1:
                        v = v.expand(bs)
                assert isinstance(v,torch.Tensor), f"class input {k} must be a tensor, an int (repeat for all samples), or a list of ints (with None being ignored)"
                assert self.use_class_emb, "classes provided but model has no class embedding"
                assert v.shape == (bs,), f"expected shape {(bs,)}, got {v.shape} for class {k}, input {v}"
                assert 0<=v.min() and v.max()<=self.class_dict[k], f"class index out of range. for class {k} expected range [0,{self.class_dict[k]}], got {v.min()} to {v.max()}"
                classes[:,list(self.class_dict.keys()).index(k)] = v
            elif torch.is_tensor(v) or isinstance(v,list):
                #here should only be image inputs
                assert self.unet_input_dict[k]["input_type"] == "image", f"input {k} is not an image input"
                exp_shape = [bs,self.unet_input_dict[k]["in_chans"],self.image_size,self.image_size]
                first_nontrivial_idx = [i for i,item in enumerate(v) if item is not None][0]
                act_shape = [len(v)]+list(v[first_nontrivial_idx].shape)
                assert act_shape == exp_shape, f"expected shape {exp_shape}, got {act_shape} for input {k}"
                bs_index = [i for i,item in enumerate(v) if item is not None]
                h[bs_index,self.unet_input_dict[k]["slice"],:,:] = torch.cat(
                                                [v[i][None] for i in bs_index]
                                                ,dim=0).type(self.inner_dtype)
        
        if timesteps.numel() == 1:
            timesteps = timesteps.expand(bs)
        assert timesteps.shape == (bs,), "timesteps must be a vector of length batch size"
        
        return h, timesteps, classes


def create_unet_from_args(args):
    if not isinstance(args,dict):
        args = copy.deepcopy(args.__dict__)
    if args["channel_multiplier"]=="auto":
        image_size = args["image_size"]
        if image_size == 256:
            if args["deeper_net"]:
                channel_mult = (1, 1, 1, 2, 2, 4, 4)
            else:
                channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            if args["deeper_net"]:
                channel_mult = (1, 1, 2, 2, 4, 4)
            else:
                channel_mult = (1, 2, 2, 4, 4)
        elif image_size == 64:
            if args["deeper_net"]:
                channel_mult = (1, 2, 3, 4, 4)
            else:
                channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 2, 2)
        elif image_size == 16:
            channel_mult = (1, 2, 2)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple([int(x) for x in args["channel_multiplier"].split(",")])
    if args["encoding_type"]=="onehot":
        out_channels = int(2**args["diff_channels"])
    elif args["encoding_type"]=="RGB":
        out_channels = 3
    else:
        out_channels=args["diff_channels"]
        if args["predict"]=="both":
            out_channels *= 2

    
    if isinstance(args["num_res_blocks"],int):
        num_res_blocks = args["num_res_blocks"]
    elif isinstance(args["num_res_blocks"],str):
        if "," in args["num_res_blocks"]:
            num_res_blocks = [int(x) for x in args["num_res_blocks"].split(",")]
        else:
            num_res_blocks = int(args["num_res_blocks"])
            
    
    if args["final_activation"].endswith("_if_x"):
        if args["predict"]=="x":
            final_act = args["final_activation"].replace("_if_x","")
        else:
            final_act = "none"
    else:
        final_act = args["final_activation"]

    mik = ModelInputKwargs(args)
    mik.construct_kwarg_table()
    unet_input_dict = mik.get_input_dict()
    unet = UNetModel(image_size=args["image_size"],
                out_channels=out_channels,
                num_res_blocks=num_res_blocks,
                model_channels=args["num_channels"],
                attention_resolutions=args["attention_resolutions"],
                dropout=args["dropout"],
                channel_mult=channel_mult,
                num_heads=args["num_heads"],
                num_heads_upsample=args["num_heads_upsample"],
                final_act=final_act,
                unet_input_dict=unet_input_dict,
                num_middle_res_blocks=args["num_middle_res_blocks"],
                )
    return unet

def get_ch_in_out(module):
    first_block_type = type(module[0]).__name__
    last_block_type = type(module[-1]).__name__
    if first_block_type in ["Downsample","Upsample","AttentionBlock","MLPBlock","ResBlock"]:
        ch_in = module[0].channels
    elif first_block_type=="Conv2d":
        ch_in = module[0].in_channels
    else:
        raise ValueError("first block type not recognized: "+first_block_type)

    if last_block_type in ["ResBlock", "Conv2d","MLPBlock"]:
        ch_out = module[-1].out_channels
    elif last_block_type in ["Downsample","Upsample"]:
        ch_out = module[-1].channels_out
    elif last_block_type=="AttentionBlock":
        ch_out = module[-1].channels
    else:
        raise ValueError("last block type not recognized: "+last_block_type)
    return ch_in, ch_out


class ModelInputKwargs:
    """
    Class to manage kwargs for model (UNet),
    as well as the dataset
    """
    def __init__(self,args,construct_args=False,assert_valid=True):
        if args is None:
            args = tp.get_args(alt_parse_args=["--model_name","analog_bits"]).__dict__
        
        if isinstance(args,Namespace):
            args = copy.deepcopy(args.__dict__)
        else:
            args = copy.deepcopy(args)
        self.args = args
        self.columns = ["name","type","spatialness","unet","support"]
        if construct_args:
            self.construct_kwarg_table()
            if assert_valid:
                self.assert_inputs_are_valid()

    def compute_hyper_params(self):
        self.hyper_params = {}
        self.hyper_params["diff_channels"] = self.args["diff_channels"]
        self.hyper_params["image_channels"] = 3

    def use_input_criteria(self):
        #a=self.args
        inputs = {
            "sample":         lambda a: True,
            "time":           lambda a: True,
            "image":          lambda a: a["p_image"]>0,
            "self_cond":      lambda a: a["p_self_cond"]>0,
            }
        return inputs
    
    def supported_inputs(self):
        unet_support = ["time","sample","image","self_cond"]
        return unet_support

    def construct_kwarg_table(self,return_df=False):
        if not hasattr(self,"hyper_params"):
            self.compute_hyper_params()
        
        im_d = {"img_size": self.args["image_size"], 
                "type": "image"}
        if self.args["encoding_type"] == "onehot":
            c_diff = int(2**self.hyper_params["diff_channels"])
        elif self.args["encoding_type"] == "RGB":
            c_diff = 3
        else:
            c_diff = self.hyper_params["diff_channels"]
        c_im = self.hyper_params["image_channels"]

        inputs = {
            "sample":         {**im_d, "in_chans": c_diff},#6X
            "image":          {**im_d, "in_chans": c_im  },#3X
            "self_cond":      {**im_d, "in_chans": c_diff},#6X
            "time":           {"type": "scalar_continuous", "min": 0.0, "max": 1.0},
            }
        #add what is needed to load each input
        load_type =  {"dynamic": [], #dynamic loading inside dataloader
                      "unique": ["time","sample","self_cond"], #unique processing required
                      "info": ["image"]#ready as-is: simply take from info
                      }
        #check that load_type is defined for all inputs
        assert_one_to_one_list_of_str(list(inputs.keys()),sum([v for v in load_type.values()],[]))
        for k,v in load_type.items():
            for k2 in v:
                inputs[k2]["load_type"] = k
        
        need_int2bit = ["sample"]
        for k in need_int2bit:
            inputs[k]["int2bit"] = True
        #add spatialness
        spatialness = {2: ["image","self_cond"], #pixelwise spatial inputs (images)
                       3: ["sample","time"]} #minimum required diffusion args
        #check that spatialness is defined for all inputs
        assert_one_to_one_list_of_str(list(inputs.keys()),sum([v for v in spatialness.values()],[]))
        for k,v in spatialness.items():
            for k2 in v:
                inputs[k2]["spatialness"] = k
        #add supports
        unet_support = self.supported_inputs()
        for k in inputs.keys():
            inputs[k]["support"] = []
            if k in unet_support:
                inputs[k]["support"].append("unet")

        input_criteria = self.use_input_criteria()

        self.kwarg_table = pd.DataFrame(columns=self.columns+["etc"])
        for name,v in inputs.items():
            append_dict = {"name": name,"etc": {}}
            for k2,v2 in v.items():
                if k2 in self.columns:
                    append_dict[k2] = v2
                else:
                    append_dict["etc"][k2] = v2
            use_input = input_criteria[name](self.args)
            append_dict["unet"] = use_input

            self.kwarg_table.loc[len(self.kwarg_table)] = append_dict

        if return_df:
            return self.kwarg_table

    def get_input_probs(self,only_nonzero=False,only_used_inputs=False,only_dynamic=True):
        probs = {k: v for k,v in self.args.items() if k.startswith("p_")}
        if only_nonzero:
            probs = {k: v for k,v in probs.items() if v>0}
        if only_used_inputs:
            used_inputs = self.kwarg_table[self.kwarg_table["unet"]]["name"]
            probs = {k: v for k,v in probs.items() if k[2:] in used_inputs}
        if only_dynamic:
            probs = {k: v for k,v in probs.items() if k[2:] in dynamic_image_keys}
        return probs

    def get_input_dict(self):
        input_dict = {}
        for row in self.kwarg_table.iterrows():
            if row[1]["unet"]:
                input_dict[row[1]["name"]] = {**row[1]["etc"],"input_type": row[1]["type"]}
        return input_dict

    def assert_inputs_are_valid(self,raise_error=True):
        try:
            assert len(self.kwarg_table)>0, "Need to construct kwarg table first, before checking validity"
            name_to_row_idx = {k: i for i,k in enumerate(self.kwarg_table["name"])}
            #assert image is an actual input
            assert len(self.kwarg_table.loc[name_to_row_idx["image"]]["support"])>=0, "Image has to be supported by unet"
            #check availability of all inputs
            for row in self.kwarg_table.iterrows():
                if row[1]["unet"]:
                    assert "unet" in row[1]["support"], "The input "+row[1]["name"]+" is not supported by the unet"
            return True
        except AssertionError as e:
            if raise_error:
                raise e
            else:
                return False

mik = ModelInputKwargs(args=None,construct_args=True,assert_valid=True)
all_input_keys = mik.kwarg_table["name"].tolist()
all_load_types = [item["load_type"] for item in mik.kwarg_table["etc"]]
dynamic_image_keys = [k for k,lt in zip(all_input_keys,all_load_types) if lt=="dynamic"]
