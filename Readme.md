# THis repository to create a Encoder from Scratch




## I AM FOLLOW THIS ARCHITECTURE
![alt text](<Images/Screenshot from 2025-08-30 16-59-00.png>)

### So this is the Encoder architecture

```
    CausalEncoder3D(
    (conv_in): CausalConv3d(
        (conv): Conv3d(3, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1))
    )
    (down_blocks): ModuleList(
        (0): CausalDownEncoder3d(
        (resnets): ModuleList(
            (0-1): 2 x CausalResnet3d(
            (norm1): GroupNorm(2, 128, eps=1e-06, affine=True)
            (conv1): CausalConv3d(
                (conv): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            )
            (norm2): GroupNorm(2, 128, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): CausalConv3d(
                (conv): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            )
            (activation_fn): SiLU()
            )
        )
        )
        (1): CausalDownEncoder3d(
        (resnets): ModuleList(
            (0): CausalResnet3d(
            (norm1): GroupNorm(2, 128, eps=1e-06, affine=True)
            (conv1): CausalConv3d(
                (conv): Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            )
            (norm2): GroupNorm(2, 256, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): CausalConv3d(
                (conv): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            )
            (activation_fn): SiLU()
            (conv_shortcut): CausalConv3d(
                (conv): Conv3d(128, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1))
            )
            )
            (1): CausalResnet3d(
            (norm1): GroupNorm(2, 256, eps=1e-06, affine=True)
            (conv1): CausalConv3d(
                (conv): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            )
            (norm2): GroupNorm(2, 256, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): CausalConv3d(
                (conv): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            )
            (activation_fn): SiLU()
            )
        )
        )
        (2): CausalDownEncoder3d(
        (resnets): ModuleList(
            (0): CausalResnet3d(
            (norm1): GroupNorm(2, 256, eps=1e-06, affine=True)
            (conv1): CausalConv3d(
                (conv): Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            )
            (norm2): GroupNorm(2, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): CausalConv3d(
                (conv): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            )
            (activation_fn): SiLU()
            (conv_shortcut): CausalConv3d(
                (conv): Conv3d(256, 512, kernel_size=(1, 1, 1), stride=(1, 1, 1))
            )
            )
            (1): CausalResnet3d(
            (norm1): GroupNorm(2, 512, eps=1e-06, affine=True)
            (conv1): CausalConv3d(
                (conv): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            )
            (norm2): GroupNorm(2, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): CausalConv3d(
                (conv): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            )
            (activation_fn): SiLU()
            )
        )
        )
        (3): CausalDownEncoder3d(
        (resnets): ModuleList(
            (0-1): 2 x CausalResnet3d(
            (norm1): GroupNorm(2, 512, eps=1e-06, affine=True)
            (conv1): CausalConv3d(
                (conv): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            )
            (norm2): GroupNorm(2, 512, eps=1e-06, affine=True)
            (dropout): Dropout(p=0.0, inplace=False)
            (conv2): CausalConv3d(
                (conv): Conv3d(512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1))
            )
            (activation_fn): SiLU()
            )
        )
        )
    )
    )

```

### The Block architecture

![alt text](<Images/Screenshot from 2025-08-30 16-52-23.png>)

## Example output 
**Below is an example of how the encoder process input data are worked.**

### example code:
```
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
```

### The channels, time, height, width 
```
        ## loop 1st [channels=128 -> 128]
        [2, 128, 8, 256, 256] -> [2, 128, 8, 256, 256]
        [2, 128, 8, 256, 256] -> [2, 128, 8, 256, 256]
                    ## downsampler=[True] stride=(1, 2, 2)
        [2, 128, 8, 256, 256] -> [2, 128, 8, 128, 128]
                    ## temporal_downsampler=[True] stride=(2, 1, 1)
        [2, 128, 8, 128, 128] -> [2, 128, 4, 128, 128]
        ---------------------------------------------------------------

        ## loop 2nd  [channels=128 -> 256]
        [2, 128, 4, 128, 128] -> [2, 256, 4, 128, 128]
        [2, 256, 4, 128, 128] -> [2, 256, 4, 128, 128]
                    ## downsampler=[True] stride=(1, 2, 2)
        [2, 256, 4, 128, 128] -> [2, 256, 4, 64, 64]
                    ## temporal_downsample=[True] stride=(2, 1, 1)
        [2, 256, 4, 64, 64] -> [2, 256, 2, 64, 64]
        --------------------------------------------------------------------

        ## loop 3rd [channels=256 -> 512]
        [2, 256, 2, 64, 64] -> [2, 512, 2, 64, 64]
        [2, 512, 2, 64, 64] -> [2, 512, 2, 64, 64]
                    ## downsampler=[True] stride=(1, 2, 2)
        [2, 512, 2, 64, 64] -> [2, 512, 2, 32, 32]
                    ## temporal_downsampler=[True] stride=(2, 1, 1)
        [2, 512, 2, 32, 32] -> [2, 512, 1, 32, 32]
        -------------------------------------------------------------------------

        ## loop 4th [channels=512 -> 512]
        [2, 512, 1, 32, 32] -> [2, 512, 1, 32, 32]
        [2, 512, 1, 32, 32] -> [2, 512, 1, 32, 32]
                    ## downsampler=[False] stride=(1, 2, 2)
        [2, 512, 1, 32, 32] -> [2, 512, 1, 32, 32]
                    ## temporal_downsampler=[False] stride=(2, 1, 1)
        [2, 512, 1, 32, 32] -> [2, 512, 1, 32, 32]
```







####  Some paper to read to required: 

- Read this paper: https://arxiv.org/pdf/2201.03545

