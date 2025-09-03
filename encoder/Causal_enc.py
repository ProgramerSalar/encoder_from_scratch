import torch 
from torch import nn 
from typing import Union, Optional, Tuple

from blocks.Causal_block import CausalDownBlock3d
from conv import CausalConv3d

class CausalEncoder3d(nn.Module):

    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, ...]] = 3,
                 stride: Union[int, Tuple[int, ...]] = 1,
                 padding: Union[int, Tuple[int, ...]] = 0,
                 encoder_types: Union[str, Tuple[str, ...]] = ("CausalDownBlock3d",),
                 block_out_channels: Tuple[int, ...] = (128,),
                 num_layers: int = 2,
                 num_groups: int = 32,
                 dropout: float = 0.0,
                 add_spatial_downsample: Tuple[bool, ...] = (True,),
                 add_temporal_downsample: Tuple[bool, ...] = (False,)
                                     
                 ):
        super().__init__()

        self.conv_in = CausalConv3d(in_channels=in_channels,
                               out_channels=block_out_channels[0],
                               kernel_size=3,
                               stride=1,
                               padding=0,
                               )
        

        # [128] -> [256]
        # [256] -> [512]
        # [512] -> [512]
        self.down_blocks = nn.ModuleList([])
        output_channels = block_out_channels[0]
        for i, encoder_type in enumerate(encoder_types):
            input_channels = output_channels            # [128], <[128], <[256], <[512]     # `>` sending sign 
            output_channels = block_out_channels[i]     # [128]>, [256]>, [512]>, [512]     # `<` accept sign
              
            down_block =  CausalDownBlock3d(in_channels=input_channels,
                              out_channels= output_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              block_out_channels=block_out_channels[i],
                              num_layers=num_layers,
                              num_groups=num_groups,
                              dropout=dropout,
                              add_spatial_downsample=add_spatial_downsample[i],
                              add_temporal_downsample=add_temporal_downsample[i])
            
            
            self.down_blocks.append(down_block)
        
            
        
    def forward(self, 
                sample: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False):
        
        sample = self.conv_in(sample,
                              is_init_image,
                              temporal_chunk)
        
        for down_block in self.down_blocks:
            sample = down_block(sample, 
                                is_init_image=is_init_image,
                                temporal_chunk=temporal_chunk)
            
        return sample
    


        
if __name__ == "__main__":

    causal_encoder3d = CausalEncoder3d(in_channels=3,
                                       out_channels=3,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       encoder_types=("CausalDownBlock3d",
                                                      "CausalDownBlock3d",
                                                      "CausalDownBlock3d",
                                                      "CausalDownBlock3d",),
                                        block_out_channels=(128, 256, 512, 512),
                                        num_layers=2,
                                        num_groups=2,
                                        add_spatial_downsample=(True, True, True, False,),
                                        add_temporal_downsample=(True, True, True, False,)
                                       )
    
    # print(causal_encoder3d)

    x = torch.randn(2, 3, 8, 256, 256)

    output = causal_encoder3d(x)
    print(output.shape)
