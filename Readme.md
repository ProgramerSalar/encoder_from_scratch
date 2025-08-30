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


####  Some paper to read to required: 

- Read this paper: https://arxiv.org/pdf/2201.03545

