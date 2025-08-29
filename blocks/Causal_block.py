import torch
from torch import nn 
import inspect
from resnets.Causal_resnet import CausalResnet3d, CausalDownsample3D, CausalTemporalDownsample3D

class CausalDownEncoder3d(nn.Module):

    def __init__(self,
                 in_channels:int = 3,
                 out_channels: int = 3,
                 num_layers: int = 3,
                 dropout: float = 0.0,
                 eps: float = 1e-6,
                 act_fn: str = "swish",
                 output_scale_factor: float = 1.0,
                 num_groups: int = 32,
                 add_spatial_downsample: bool = True,
                 add_temporal_downsample: bool = False,
                 ):
        
        super().__init__()
        self.add_spatial_downsample = add_spatial_downsample
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.add_temporal_downsample = add_temporal_downsample
        
        resnets= []
        for i in range(num_layers):
        
            input_channels = in_channels if i == 0 else out_channels 
            output_channels = out_channels

            # print(f"what is the index: {i}, input_channel: {input_channels} and output_channels: {output_channels}")

            resnets.append(CausalResnet3d(in_channels=input_channels,
                               out_channels=output_channels,
                                num_groups=num_groups,
                                eps=eps,
                                act_fn=act_fn,
                                output_scale_factor=output_scale_factor,
                                dropout=dropout,
                               ))
            
        self.resnets = nn.ModuleList(resnets)

        
        if self.add_spatial_downsample:

            self.downsamplers = nn.ModuleList([
                CausalDownsample3D(in_channels=in_channels,
                                   out_channels=out_channels,
                                   use_conv=True)
            ])

        else:
            self.downsamplers = None


        if self.add_temporal_downsample:

            self.temporal_downsampler = nn.ModuleList([
                CausalTemporalDownsample3D()
            ])
        


               




    def forward(self, 
                sample: torch.FloatTensor,
                temb=None,
                is_init_image=True,
                temporal_chunk=False) -> torch.FloatTensor:
        


        for resnet in self.resnets:
            
            hidden_states = resnet(sample,
                                   temb=temb,
                                   is_init_image=is_init_image,
                                   temporal_chunk=temporal_chunk)
            
            
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states,
                                            is_init_image=is_init_image,
                                            temporal_chunk=temporal_chunk)
                


        

            
        return hidden_states




if __name__ == "__main__":

    causal_down_encoder_3d = CausalDownEncoder3d(in_channels=64,
                                                 out_channels=64,
                                                 num_layers=2)
    print(causal_down_encoder_3d)
    print('-'*20)


    x = torch.randn(2, 64, 8, 256, 256)

    output = causal_down_encoder_3d(x)
    print(output.shape)
    # pass