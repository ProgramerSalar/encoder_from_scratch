import torch 
from torch import nn 
from typing import Tuple

from conv import CausalConv3d
from blocks.Causal_block import CausalDownEncoder3d

class CausalEncoder3D(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 down_block_type: Tuple[str, ...] = ("DownEncoderBlockCausal3D",),
                 block_out_channels: Tuple[int, ...] = (128,),
                 num_layers: int = 3,
                 dropout: float = 0.0,
                 eps: float = 1e-6,
                 act_fn: str = "swish",
                 output_scale_factor: float = 1.0,
                 num_groups: int = 32,
                 add_spatial_downsample: Tuple[bool, ...] = (True,),
                 add_temporal_downsample: Tuple[bool, ...] = (True,)
                 ):
        super().__init__()
        
        # [3] -> [64]
        self.conv_in = CausalConv3d(in_channels=in_channels,
                                    out_channels=block_out_channels[0])
        

        # mid block 
        output_channels = block_out_channels[0]
        self.down_blocks = nn.ModuleList([])
        for i, down_block_type in enumerate(down_block_type):
            input_channels = output_channels
            output_channels = block_out_channels[i]
            # [64] -> [64]
            # [64] -> [128]
            # [128] -> [256] 
            # [256] -> [256]
            down_block = CausalDownEncoder3d(
                            in_channels=input_channels,
                            out_channels=output_channels,
                            num_layers=num_layers[i],
                            dropout=dropout,
                            eps=eps,
                            act_fn=act_fn,
                            output_scale_factor=output_scale_factor,
                            num_groups=num_groups,
                            add_spatial_downsample=add_spatial_downsample[i],
                            add_temporal_downsample=add_temporal_downsample
            )
            self.down_blocks.append(down_block)

    def forward(self, 
                x,
                is_init_image=True,
                temporal_chunk=False):

        # [3] -> [128]
        sample = self.conv_in(x,
                              is_init_image=is_init_image,
                              temporal_chunk=temporal_chunk)
        
        for down_block in self.down_blocks:
            sample = down_block(sample, 
                                is_init_image=is_init_image,
                                temporal_chunk=temporal_chunk)
            
        return sample      



if __name__ == "__main__":
    
    causal_encoder_3d = CausalEncoder3D(in_channels=3,
                                        out_channels=3,
                                        down_block_type=("DownEncoderBlockCausal3D",
                                                         "DownEncoderBlockCausal3D",
                                                         "DownEncoderBlockCausal3D",
                                                         "DownEncoderBlockCausal3D"),
                                        block_out_channels=(128, 256, 512, 512),
                                        num_layers=(2, 2, 2, 2),
                                        num_groups=2,
                                        add_spatial_downsample=(True, True, True, False),
                                        add_temporal_downsample=(True, True, True, False)
                                        )
    

    # print(causal_encoder_3d)

    print('-'*20)
    x = torch.randn(2, 3, 8, 256, 256)
    output = causal_encoder_3d(x)
    # torch.Size([2, 3, 8, 256, 256]) -> torch.Size([2, 512, 1, 32, 32])
    print(f"what is the encoder output shape: {output.shape}")
