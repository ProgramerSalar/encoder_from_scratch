import torch
from torch import nn 

from resnets.Causal_resnet import CausalResnet3d

class CausalDownEncoder3d(nn.Module):

    def __init__(self,
                 in_channels:int = 3,
                 out_channels: int = 3,
                 num_layers: int = 3,
                 dropout: float = 0.0,
                 eps: float = 1e-6,
                 act_fn: str = "swish",
                 output_scale_factor: float = 1.0,
                 num_groups: int = 32
                 ):
        
        super().__init__()
        
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels 
            output_channels = out_channels

            # print(f"input_channel: {input_channels} and output_channels: {output_channels}")

            self.resnets = nn.ModuleList([
                CausalResnet3d(in_channels=input_channels,
                               out_channels=output_channels,
                                num_groups=num_groups,
                                eps=eps,
                                act_fn=act_fn,
                                output_scale_factor=output_scale_factor,
                                dropout=dropout,
                                
                               )
            ])


    def forward(self, 
                sample: torch.FloatTensor,
                temb=None,
                is_init_image=True,
                temporal_chunk=False) -> torch.FloatTensor:
        


        for resnet in self.resnets:
            hidden_states = resnet(sample,
                                   temb=None,
                                   is_init_image=is_init_image,
                                   temporal_chunk=temporal_chunk)
            
        return hidden_states




if __name__ == "__main__":

    causal_down_encoder_3d = CausalDownEncoder3d(in_channels=64,
                                                 out_channels=64)
    print(causal_down_encoder_3d)


    x = torch.randn(2, 64, 8, 256, 256)

    output = causal_down_encoder_3d(x)
    print(output.shape)