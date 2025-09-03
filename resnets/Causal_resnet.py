import torch
from torch import nn 

from conv import CausalConv3d, CausalGroupNorm

class CausalResnet3d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_groups: int = 32,
                 dropout: float = 1e-6,
                 use_in_shortcut: bool = None
                 ):
        super().__init__()

        
        # [128]
        self.norm1 = CausalGroupNorm(num_groups=num_groups,
                                     num_channels=in_channels)
        
        self.conv1 = CausalConv3d(in_channels=in_channels,
                                  out_channels=out_channels)
        

        self.norm2 = CausalGroupNorm(num_groups=num_groups,
                                     num_channels=out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        
        conv_2d_out_channels = out_channels
        self.conv2 = CausalConv3d(in_channels=out_channels,
                                  out_channels=out_channels)
        
        self.activation_fn = nn.SiLU()

        # this is true when in_channels != conv_2d_out_channels 
        # [128] != [256] => True 
        # [256] != [512] => True 
        if use_in_shortcut is None:
            self.use_in_shortcut = in_channels != conv_2d_out_channels

        else:
            self.use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = CausalConv3d(in_channels=in_channels,
                                              out_channels=conv_2d_out_channels,
                                              kernel_size=1,
                                              stride=1,
                                              bias=True
                                              )
            
        else:
            self.use_in_shortcut = False


        
    def forward(self, 
                x,
                is_init_image=True,
                temporal_chunk=False):
        

        input_tensor = x 
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.conv1(x, is_init_image, temporal_chunk)

        x = self.norm2(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.conv2(x, is_init_image, temporal_chunk)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor, is_init_image, temporal_chunk)

        output_tensor = (input_tensor + x) / 1.0

        return output_tensor





class CausalDownSample2x(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 ):
        
        super().__init__()

        stride = (1, 2, 2)
        
        self.conv = CausalConv3d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 bias=True)

    def forward(self, 
                hidden_size: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False):
        
        hidden_size = self.conv(hidden_size,
                      is_init_image=is_init_image,
                      temporal_chunk=temporal_chunk)
        

        return hidden_size
    




class CausalTemporalDownsampele2x(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3):
        
        super().__init__()
        stride = (2, 1, 1)
        self.in_channels = in_channels

        self.conv = CausalConv3d(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=kernel_size,
                                 stride=stride)
        

    def forward(self,
                hidden_state: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False):
        
        assert hidden_state.shape[1] == self.in_channels, 'make sure `channels` are equal to `in_channels`!'
        hidden_state = self.conv(hidden_state,
                                 is_init_image=is_init_image,
                                 temporal_chunk=temporal_chunk)
        
        return hidden_state
    


    


if __name__ == "__main__":

    # causal_downsample_2x = CausalDownSample2x(in_channels=128,
    #                                           out_channels=128)
    
    # x = torch.randn(2, 128, 8, 256, 256)

    # output = causal_downsample_2x(x)
    # print(output.shape)
    # --------------------------------------------------------------------------------
    causal_temporaldownsampele_2x = CausalTemporalDownsampele2x(in_channels=128,
                                              out_channels=256)
    
    x = torch.randn(2, 128, 8, 256, 256)

    output = causal_temporaldownsampele_2x(x)
    print(output.shape)