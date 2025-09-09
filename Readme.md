# THis repository to create a Encoder from Scratch




## I AM FOLLOW THIS ARCHITECTURE
![alt text](<Images/Screenshot from 2025-08-30 16-59-00.png>)



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








####  Some paper to read to required: 

- Read this paper: https://arxiv.org/pdf/2201.03545

