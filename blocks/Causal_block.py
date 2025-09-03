import torch
from torch import nn 
from typing import Union, Tuple
from diffusers.models.attention_processor import Attention
from einops import rearrange


from resnets.Causal_resnet import CausalResnet3d, CausalDownSample2x, CausalTemporalDownsampele2x

class CausalDownBlock3d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]] = 3,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 block_out_channels: int = 128,
                 num_layers: int = 3,
                 num_groups: int = 32,
                 dropout: float = 0.0,
                 add_spatial_downsample: bool = True,
                 add_temporal_downsample: bool = False
                 ):
        
        super().__init__()

        
        self.resnets = nn.ModuleList([])
        for num_layer in range(num_layers):
            input_channels = in_channels if num_layer == 0 else out_channels
            # print(f"[Block] what is the shape of input_channels: {in_channels} and output_channels: {out_channels}")

            resnet = CausalResnet3d(in_channels=input_channels,
                           out_channels=out_channels,
                           num_groups=num_groups,
                           dropout=dropout)
            
            self.resnets.append(resnet)

            
        if add_spatial_downsample:
            
            self.downsamplers = nn.ModuleList([
                CausalDownSample2x(in_channels=out_channels,
                                   out_channels=out_channels,
                                   kernel_size=kernel_size,
                                   )
            ])

        else:
            self.downsamplers = None

        if add_temporal_downsample:
            self.temporal_downsampler = nn.ModuleList([
                CausalTemporalDownsampele2x(in_channels=out_channels,
                                            out_channels=out_channels,
                                            kernel_size=kernel_size)
            ])
        else:
            self.temporal_downsampler = None

    def forward(self,
                hidden_size: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False):
        

        for resnet in self.resnets:
            hidden_size = resnet(hidden_size,
                                 is_init_image=is_init_image,
                                 temporal_chunk=temporal_chunk)
            
            
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_size = downsampler(hidden_size, is_init_image, temporal_chunk)
                

        if self.temporal_downsampler is not None:
            for temporal_downsample in self.temporal_downsampler:   
                hidden_size = temporal_downsample(hidden_size, is_init_image, temporal_chunk)
             

        return hidden_size
    

class CausalMidBlock3d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 attention_head_dim: int = 512,
                 num_groups: int = 32,
                 add_attention: bool = True,
                 dropout: float = 0.0,
                 num_layers: int = 1,
                 ):
        
        super().__init__()
        self.add_attention = add_attention

        resnets = [
            CausalResnet3d(
                in_channels=in_channels,
                out_channels=in_channels,
                num_groups=num_groups,
                dropout=dropout,
            )
        ]
        
        attentions = []
        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                    query_dim=in_channels,
                    heads=in_channels // attention_head_dim,
                    dim_head=attention_head_dim,
                    rescale_output_factor=1.0,
                    eps=1e-6,
                    norm_num_groups=num_groups,
                    spatial_norm_dim=None,
                    residual_connection=True,
                    bias=True,
                    upcast_softmax=True,
                    _from_deprecated_attn_block=True
                )
                )
            else:
                attentions.append(None)

            resnets.append(
                CausalResnet3d(
                in_channels=in_channels,
                out_channels=in_channels,
                num_groups=num_groups,
                dropout=dropout,
            )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        
        


    def forward(self, 
                hidden_size: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False):
        
        hidden_size = self.resnets[0](hidden_size, is_init_image, temporal_chunk)

        # [batch_size, channels, time, height, width]
        t = hidden_size.shape[2]

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_size = rearrange(hidden_size, 'b c t h w -> b t c h w')
                hidden_size = rearrange(hidden_size, 'b t c h w -> (b t) c h w')
                hidden_size = attn(hidden_size)
                hidden_size = rearrange(hidden_size, '(b t) c h w -> b t c h w', t=t)
                hidden_size = rearrange(hidden_size, 'b t c h w -> b c t h w')

            hidden_size = resnet(hidden_size, is_init_image, temporal_chunk)
        
        return hidden_size

        
        return hidden_size