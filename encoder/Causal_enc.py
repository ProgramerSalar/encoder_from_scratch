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
                 block_out_channels: Tuple[int, ...] = (64,),
                 num_layers: int = 3,
                 dropout: float = 0.0,
                 eps: float = 1e-6,
                 act_fn: str = "swish",
                 output_scale_factor: float = 1.0,
                 num_groups: int = 32
                 ):
        

        super().__init__()
        
        # [3] -> [64]
        self.conv_in = CausalConv3d(in_channels=in_channels,
                                    out_channels=block_out_channels[0])
        

        # mid block 
        output_channels = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_type):
            input_channels = output_channels
            output_channels = block_out_channels[i]

            print(f"what is the input_channels: {input_channels} and output_channels: {output_channels}")
            # [64] -> [64]
            # [64] -> [128]
            # [128] -> [256] 
            # [256] -> [256]
            self.down_blocks = nn.ModuleList([CausalDownEncoder3d(
                            in_channels=input_channels,
                            out_channels=output_channels,
                            num_layers=num_layers,
                            dropout=dropout,
                            eps=eps,
                            act_fn=act_fn,
                            output_scale_factor=output_scale_factor,
                            num_groups=num_groups
            )])


    def forward(self, 
                x,
                is_init_image=True,
                temporal_chunk=False):

        sample = self.conv_in(x,
                              is_init_image=is_init_image,
                              temporal_chunk=temporal_chunk)
        
        for down_block in self.down_blocks:
            sample = down_block(sample,
                                is_init_image=is_init_image,
                                temb=None,
                                temporal_chunk=temporal_chunk)
            

        return sample      



if __name__ == "__main__":
    
    causal_encoder_3d = CausalEncoder3D(in_channels=3,
                                        out_channels=3,
                                        down_block_type=("DownEncoderBlockCausal3D",
                                                         "DownEncoderBlockCausal3D",
                                                         "DownEncoderBlockCausal3D",
                                                         "DownEncoderBlockCausal3D"),
                                        block_out_channels=(64, 128, 256, 256),
                                        num_groups=2
                                        )
    

    print(causal_encoder_3d)

    # x = torch.randn(2, 3, 8, 256, 256)
    # output = causal_encoder_3d(x)
    # print(output.shape)