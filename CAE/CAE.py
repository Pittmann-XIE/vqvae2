import torch
from torch import nn
from transformers import CLIPVisionModel

class Encoder(nn.Module):
    # Default backbone to patch16 to generate exactly 196 patches for a 224x224 image
    def __init__(self, embed_dim=256, num_heads=8, pretrained_model_name="openai/clip-vit-base-patch16"):
        super().__init__()
        
        # The Pretrained Vision Backbone (CLIP) [cite: 536]
        self.image_processor = CLIPVisionModel.from_pretrained(pretrained_model_name, use_safetensors=True)
        
        # Freeze the backbone parameters
        for param in self.image_processor.parameters():
            param.requires_grad = False
            
        clip_hidden_size = self.image_processor.config.hidden_size 
        
        self.feature_projection = nn.Linear(clip_hidden_size, embed_dim)
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, dynamic_queries, condition_embedding, pixel_values):
        # pixel_values shape: [Batch, 3, 224, 224] 
        with torch.no_grad():
            vision_outputs = self.image_processor(pixel_values=pixel_values)
            raw_image_features = vision_outputs.last_hidden_state
            
        # Strip the [CLS] token -> Shape becomes [Batch, 196, 768]
        patch_features = raw_image_features[:, 1:, :] 
        
        # Project down -> Shape becomes [Batch, 196, 256]
        projected_features = self.feature_projection(patch_features) 
        
        # Inject the combined task & ratio semantics
        q = dynamic_queries + condition_embedding 
        
        # Cross-Attend -> Output shape: [Batch, K, 256]
        z, _ = self.cross_attention(query=q, key=projected_features, value=projected_features)
        z_final = self.layer_norm(z + q)
        
        return z_final
    
class ConditionEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        # Encodes the categorical task (Pose vs Recon)
        self.task_embed = nn.Embedding(num_embeddings=2, embedding_dim=embed_dim)
        # Encodes the continuous ratio float (0.1 to 1.0)
        self.ratio_proj = nn.Linear(1, embed_dim)
        
    def forward(self, task_id, target_ratio_tensor):
        t_emb = self.task_embed(task_id).unsqueeze(1) # [Batch, 1, 256]
        r_emb = self.ratio_proj(target_ratio_tensor).unsqueeze(1) # [Batch, 1, 256]
        
        # The queries now know WHAT to look for AND HOW MUCH bandwidth they have
        return t_emb + r_emb

class RatioSpecificQueryGenerator(nn.Module):
    def __init__(self, max_queries=196, embed_dim=256, num_bins=10):
        super().__init__()
        self.max_queries = max_queries
        self.num_bins = num_bins
        
        # 10 separate "models" (represented here as 10 distinct query pools)
        # Shape: [10, 196, 256]
        self.query_bank = nn.Parameter(torch.randn(num_bins, max_queries, embed_dim))
        
    def forward(self, target_ratio: float, batch_size: int):
        # 1. Discretize target_ratio into 10 values (Bins 0 to 9)
        bin_idx = min(int((target_ratio - 0.001) * self.num_bins), self.num_bins - 1)
        
        # 2. Calculate how many tokens (K) this ratio allows
        K = max(1, int(self.max_queries * target_ratio))
        
        # 3. Fuse/Select the specific weights for this ratio
        queries = self.query_bank[bin_idx, :K, :].expand(batch_size, -1, -1)
        
        return queries

class ImageReconstructionDecoder(nn.Module):
    def __init__(self, embed_dim=256, num_layers=4, num_heads=8, output_size=224, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = output_size // patch_size  # 14
        self.num_patches = self.grid_size ** 2      # 196
        
        # 1. Learnable Spatial Queries (representing the 14x14 grid)
        self.spatial_queries = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        # 2. Positional Embeddings for the Decoder
        # Without these, the Transformer doesn't strictly know the 2D layout of the patches
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        # 3. Stacked Transformer Decoder Layers
        # This allows the model to iteratively refine the image reconstruction
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4, # Higher capacity for texture detail
                dropout=0.1,
                activation="gelu",
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # 4. Pixel Head: Maps the 256-dim embedding to a 16x16x3 RGB patch
        pixels_per_patch = patch_size * patch_size * 3
        self.pixel_predictor = nn.Linear(embed_dim, pixels_per_patch)
        
    def forward(self, z, condition_embedding):
        batch_size = z.size(0)
        
        # Initialize queries and add positional information
        # The condition_embedding (task + ratio) is fused here to guide the reconstruction
        x = self.spatial_queries.expand(batch_size, -1, -1) + self.pos_embed
        
        # The 'memory' for the decoder is the encoder's output (z) fused with task context
        memory = z + condition_embedding
        
        # Iterative refinement through the ViT layers
        for layer in self.layers:
            x = layer(tgt=x, memory=memory)
            
        x = self.final_norm(x)
        
        # Project tokens to pixel patches
        patches = self.pixel_predictor(x) # [Batch, 196, 768]
        
        # Reshape patches back into a full image [Batch, 3, 224, 224]
        patches = patches.view(batch_size, self.grid_size, self.grid_size, 3, self.patch_size, self.patch_size)
        image = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        image = image.view(batch_size, 3, 224, 224)
        
        return image


class PoseEstimationHead(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_keypoints=17):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.keypoint_queries = nn.Parameter(torch.randn(1, num_keypoints, embed_dim))
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        self.coord_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 3) 
        )
        
    def forward(self, z):
        batch_size = z.size(0)
        q = self.keypoint_queries.expand(batch_size, -1, -1) # [Batch, 17, 256]
        
        kp_features, _ = self.cross_attention(query=q, key=z, value=z)
        kp_features = self.layer_norm(kp_features + q) 
        
        raw_keypoints = self.coord_predictor(kp_features) 
        yolo_keypoints = torch.sigmoid(raw_keypoints) # [Batch, 17, 3]
        
        return yolo_keypoints

class ConditionalAutoEncoder(nn.Module):
    def __init__(self, hidden_dim=256, max_image_patches=196):
        super().__init__()
        # Use ConditionEncoder to handle both Task and Ratio
        self.condition_encoder = ConditionEncoder(embed_dim=hidden_dim)
        self.query_generator = RatioSpecificQueryGenerator(max_queries=max_image_patches, embed_dim=hidden_dim)
        self.vision_task_encoder = Encoder(embed_dim=hidden_dim)
        
        self.taskhead_humanpose = PoseEstimationHead(embed_dim=hidden_dim, num_keypoints=17)
        self.decoder_imagerecon = ImageReconstructionDecoder(embed_dim=hidden_dim)
    
    def forward(self, pixel_values, task_name: str, target_ratio: float):
        # Expecting pixel_values to be strictly [Batch, 3, 224, 224]
        batch_size = pixel_values.size(0)
        device = pixel_values.device
        
        task_id = torch.tensor(
            [0 if task_name == 'pose' else 1], 
            dtype=torch.long, 
            device=device
        ).expand(batch_size)
        
        # Format the ratio into a tensor
        ratio_tensor = torch.tensor([[target_ratio]], dtype=torch.float32, device=device).expand(batch_size, -1)
        
        # Get the combined task + ratio embedding
        condition_embedding = self.condition_encoder(task_id, ratio_tensor)
        dynamic_queries = self.query_generator(target_ratio, batch_size)
        
        # Encode
        z = self.vision_task_encoder(dynamic_queries, condition_embedding, pixel_values)
        
        # Branch
        if task_name == 'pose':
            results = self.taskhead_humanpose(z)
        elif task_name == 'recon':
            results = self.decoder_imagerecon(z, condition_embedding) 
        else:
            raise ValueError(f"Unknown task: {task_name}")
            
        return results, z
    




# class ImageReconstructionDecoder(nn.Module):
#     def __init__(self, embed_dim=256, num_heads=8, output_size=224, patch_size=16, in_channels=3):
#         super().__init__()
#         self.output_size = output_size
#         self.patch_size = patch_size
#         self.grid_size = output_size // patch_size # 224 // 16 = 14
#         self.num_patches = self.grid_size ** 2     # 14^2 = 196
        
#         # Shape: [1, 196, 256]
#         self.spatial_queries = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
#         self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
#         self.layer_norm = nn.LayerNorm(embed_dim)
        
#         pixels_per_patch = patch_size * patch_size * in_channels # 16 * 16 * 3 = 768
#         self.pixel_predictor = nn.Linear(embed_dim, pixels_per_patch)
        
#     def forward(self, z, condition_embedding):
#         batch_size = z.size(0)
#         q = self.spatial_queries.expand(batch_size, -1, -1) # [Batch, 196, 256]
        
#         # Decoder is also aware of the combined task/ratio condition
#         k_v = z + condition_embedding
        
#         decoded_tokens, _ = self.cross_attention(query=q, key=k_v, value=k_v)
#         decoded_tokens = self.layer_norm(decoded_tokens + q) # [Batch, 196, 256]
        
#         patches = self.pixel_predictor(decoded_tokens) # [Batch, 196, 768]
        
#         B, N, P = patches.shape
#         patches = patches.view(B, self.grid_size, self.grid_size, 3, self.patch_size, self.patch_size)
#         image = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
#         image = image.view(B, 3, self.output_size, self.output_size) # [Batch, 3, 224, 224]
        
#         return image
