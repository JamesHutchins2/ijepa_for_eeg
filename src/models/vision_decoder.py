import sys
import torch
import torch.nn as nn
import numpy as np
from timm.models.vision_transformer import Block
import torch.nn.functional as F
import src.models.utils as ut
import math


class vision_decoder(nn.Module):
    """
    A Masked Autoencoder for 1D data (e.g., time series), using a transformer-based architecture.
    
    This model is designed to encode 1D input data into a lower-dimensional space and then decode 
    it back to its original dimension, with a focus on reconstructing the original data from 
    partial (masked) inputs. It features a Vision Transformer (ViT) backbone for both encoding and 
    decoding processes.
    
    Parameters:
    - time_len: Length of the input time series.
    - patch_size: Size of each patch into which the input data is divided.
    - embed_dim: Dimensionality of the embedding space for the encoder.
    - in_chans: Number of input channels.
    - Various parameters for configuring the depth and heads of the transformer model, along with other hyperparameters.
    """
    
    def __init__(self, time_len=486, patch_size=4, embed_dim=1280, in_chans=128,
                 depth=6, num_heads=3, decoder_embed_dim=786, 
                 decoder_depth=2, decoder_num_heads=1,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()

        # Initialize the encoder part of the MAE
        # This involves embedding the input patches and applying transformer blocks to them
        
        num_patches = 55
        self.num_patches = num_patches
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)


        # Initialize the decoder part of the MAE
        # It decodes the encoded features back to the original data dimensionality
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 44))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)

        


        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, in_chans * patch_size)

        # Store some parameters and initializations
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.initialize_weights()
    def interpolate_pos_encoding(self, x, pos_embed):
        
        print(f"at positional encoding interpolation, x shape: {x.shape}, pos_embed shape: {pos_embed.shape}")
        
        
        npatch = x.shape[1] - 1
        print(f"npatch: {npatch}")
        N = pos_embed.shape[1] - 1
        print(f"N: {N}")
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]

        #print out all of the shapes

        
        print(f"X shape: {x.shape}")
        print(f"Pos emb shape: {pos_embed.shape}")

        print(f"shape of pos after reshape will be: 1, {int(math.sqrt(N))}, {int(math.sqrt(N))}, {dim}")

        print(f"scale factor: {math.sqrt(npatch / N)}")
        


        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)

    def initialize_weights(self):
        """
        Initializes the weights of the model, setting up specific initial values for different types of layers.
        This includes setting up the positional embeddings with a sin-cos pattern, initializing weights for the patch embedding,
        class token, mask token, and standard layers (Linear, LayerNorm, Conv1d) following best practices.
        """
        
        # Initialize positional embeddings with sin-cos pattern for encoder and decoder
        # This uses a utility function to generate the embeddings, assuming it creates embeddings suitable for 1D data
        pos_embed = ut.get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches, cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = ut.get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.num_patches, cls_token=False)
        print("shaping of the decoder-pos-embed: ")
        print("dimension 1: " + str(decoder_pos_embed.shape[-1]))
        print("dimension 2: " + str(self.num_patches))
        
        
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # Initialize the patch embedding weights similar to nn.Linear's initialization method
        #w = self.patch_embed.proj.weight.data
        #torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize class and mask tokens with normal distribution
        
        torch.nn.init.normal_(self.mask_token, std=.02)

        # Apply custom initialization to all layers in the model
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Applies custom initialization for different types of layers within the model.
        """
        if isinstance(m, nn.Linear):
            # Initialize Linear layers with Xavier uniform distribution
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                # Set biases to zero
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            # Initialize LayerNorm layers with biases set to zero and weights set to one
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            # Initialize Conv1d layers with normal distribution for weights
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                # Set biases to zero for Conv1d layers
                nn.init.constant_(m.bias, 0)

    
    

    def forward_decoder(self, x, ids_restore):
        """
        Decodes the encoded representation back into the original data space.

        Args:
        - x (Tensor): Encoded data.
        - ids_restore (Tensor): Indices to restore the original ordering of the sequence.

        Returns:
        - The decoded representation of the data.
        """
        print("decoder called")
        x = self.decoder_embed(x)  # Embed decoded tokens
        print("XSHAPE: " + str(x.shape))
        print("IDS SHAPE: " + str(ids_restore.shape))
        ids_restore = ids_restore.unsqueeze(-1)
        
        #x = torch.cat((ids_restore, x), dim=2)
        
        print(f"x with ids_restore: {x.shape}")
        # add ids_restore to x
        
        try:
            positional_embedding = self.interpolate_pos_encoding(x, self.decoder_pos_embed)
            print(f"shape of x: {x.shape}, shape of the positional embeddings: {positional_embedding.shape}")
            x = x + positional_embedding # Add positional embeddings
        except Exception as e:
            print(f'Encountered exception in decoder {e}')
        print("positional embeddings added")
        # Process through decoder transformer blocks
        for blk in self.decoder_blocks:
            print('block call')
            x = blk(x)
            
        x = self.decoder_norm(x)
        print("normalized")
        x = self.decoder_pred(x)  # Project back to the original data space
        print('decoder predictor run')
        x = x[:, 1:, :]  # Remove class token for the final output

        
        print("forward_decoder is complete, outputting shape of : ", str(x.shape))
        return x
    
    

    def forward(self, latent, ids_restore):
        
        print("mask tokens shape: " + str(ids_restore[0].shape))
        
        ids_restore = ids_restore[0]
        
        print("predicting...")
        print("given latent shape: " + str(latent.shape))
        
        pred = self.forward_decoder(latent,ids_restore) 
        
        
        return  pred, 

