import torch
from torch import nn 


class ResnetBlock2d(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_groups: int = 32,
                 dropout: float = 0.0,
                 use_in_shortcut: bool = None):
        
        super().__init__()

        self.norm1 = nn.GroupNorm(num_channels=in_channels,
                                  num_groups=num_groups,
                                  eps=1e-6,
                                  affine=True)
        self.conv1 = nn.Conv3d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        
        self.norm2 = nn.GroupNorm(num_channels=out_channels,
                                  num_groups=num_groups,
                                  eps=1e-6,
                                  affine=True)
        
        self.dropout = nn.Dropout(dropout)
        conv_2d_out_channels = out_channels
        self.conv2 = nn.Conv3d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.activation_fn = nn.SiLU()

        self.conv_shortcut = None
        if use_in_shortcut is None:
            self.use_in_shortcut = in_channels != conv_2d_out_channels
            self.conv_shortcut = nn.Conv3d(in_channels=in_channels,
                                           out_channels=conv_2d_out_channels,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           bias=True)

        else:
            self.use_in_shortcut

    
    def forward(self,
                x: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False):
        
        input_tensor = x 
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.conv2(x)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = (input_tensor + x) / 1.0 

        return output_tensor
         

            
            

