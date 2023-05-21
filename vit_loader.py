from efficientViT import ViT
from simpleViT import simple_ViT
from linformer import Linformer


def vit_loader(args):
    if(args == "simple"):
        
        model = simple_ViT(
            image_size = 256, # 256 Image size. If you have rectangular images, make sure your image size is the maximum of the width and height
            patch_size = 16, # 16 Number of patches. image_size must be divisible by patch_size. The number of patches is:  n = (image_size // patch_size) ** 2 and n must be greater than 16.
            num_classes = 200, #3, 200 Number of classes to classify.
            dim = 512, #1024, Last dimension of output tensor after linear transformation nn.Linear(..., dim).
            depth = 3, # 6 Number of Transformer blocks.
            heads = 8, # 16 Number of heads in Multi-head Attention layer.
            mlp_dim = 1024 # 2048 Dimension of the MLP (FeedForward) layer.
        )
    
    elif(args == "efficient"):
        """
        An implementation of Linformer in Pytorch. Linformer comes with two deficiencies. (1) It does not work for the auto-regressive case. (2) Assumes a fixed sequence length. However, if benchmarks show it to perform well enough, it will be added to this repository as a self-attention layer to be used in the encoder.

        Linformer has been put into production by Facebook!
        """
        # input PReLU() instead of GELU()
        efficient_transformer = Linformer(
            dim=128,
            seq_len=64+1,  # 8x8 patches + 1 cls-token
            depth=12,
            heads=8,
            k=64
        )

        model = ViT(
            dim=128, # 128
            image_size=256, # 224
            patch_size=32, # 32
            num_classes=200, # 2
            transformer=efficient_transformer, # efficient_transformer
            channels=3, # 3
        )
    return model   
            
