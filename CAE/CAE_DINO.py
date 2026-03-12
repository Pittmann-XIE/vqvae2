import torch
from torch import nn
from transformers import AutoModel

class Encoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, pretrained_model_name="facebook/dinov3-vitb16-pretrain-lvd1689m", use_precomputed_dino=False):
        super().__init__()
        self.use_precomputed_dino = use_precomputed_dino
        
        if not self.use_precomputed_dino:
            # Load the Pretrained Vision Backbone (DINOv3)
            self.backbone = AutoModel.from_pretrained(pretrained_model_name)
            # Freeze the backbone parameters
            for param in self.backbone.parameters():
                param.requires_grad = False
            hidden_size = self.backbone.config.hidden_size 
        else:
            self.backbone = None
            hidden_size = 768 # Hardcoded hidden size for DINOv3 ViT-Base
            
        self.feature_projection = nn.Linear(hidden_size, embed_dim)
        self.cls_projection = nn.Linear(hidden_size, embed_dim)
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, dynamic_queries, condition_embedding, pixel_values=None, precomputed_features=None):
        if not self.use_precomputed_dino:
            with torch.no_grad():
                vision_outputs = self.backbone(pixel_values=pixel_values)
                raw_image_features = vision_outputs.last_hidden_state
        else:
            if precomputed_features is None:
                raise ValueError("Model is set to use precomputed features, but none were provided.")
            raw_image_features = precomputed_features
            
        # DINOv3 Sequence: 1 [CLS] + 4 [Register Tokens] + N [Patch Tokens]
        if not self.use_precomputed_dino:
            # Registers are present
            cls_token_raw = raw_image_features[:, 0:1, :]       
            patch_features_raw = raw_image_features[:, 5:, :]   
        else:
            # Registers were already stripped during preprocessing
            cls_token_raw = raw_image_features[:, 0:1, :]       
            patch_features_raw = raw_image_features[:, 1:, :]   # Start at index 1 to get all 432 patches
        
        projected_features = self.feature_projection(patch_features_raw) # [Batch, 432, 256]
        projected_cls = self.cls_projection(cls_token_raw)               #[Batch, 1, 256]
        
        q = dynamic_queries + condition_embedding 
        
        z, _ = self.cross_attention(query=q, key=projected_features, value=projected_features)
        z_final = self.layer_norm(z + q)
        
        return z_final, projected_cls
    
class ConditionEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.task_embed = nn.Embedding(num_embeddings=2, embedding_dim=embed_dim)
        self.ratio_proj = nn.Linear(1, embed_dim)
        
    def forward(self, task_id, target_ratio_tensor):
        t_emb = self.task_embed(task_id).unsqueeze(1) # [Batch, 1, 256]
        r_emb = self.ratio_proj(target_ratio_tensor).unsqueeze(1) # [Batch, 1, 256]
        return t_emb + r_emb

class RatioSpecificQueryGenerator(nn.Module):
    # Updated max_queries to 432 to match the new 16x27 patch grid
    def __init__(self, max_queries=432, embed_dim=256, num_bins=10):
        super().__init__()
        self.max_queries = max_queries
        self.num_bins = num_bins
        self.query_bank = nn.Parameter(torch.randn(num_bins, max_queries, embed_dim))
        
    def forward(self, target_ratio: float, batch_size: int):
        bin_idx = min(int((target_ratio - 0.001) * self.num_bins), self.num_bins - 1)
        K = max(1, int(self.max_queries * target_ratio))
        queries = self.query_bank[bin_idx, :K, :].expand(batch_size, -1, -1)
        return queries

class DCAEUpsampleBlock(nn.Module):
    """
    DC-AE Block using Channel-to-Space (PixelShuffle) and Channel Duplicating.
    """
    def __init__(self, in_channels, out_channels, upscale_factor=2):
        super().__init__()
        # To maintain 'out_channels' after PixelShuffle(2), we need out_channels * 4 internally
        self.conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class ImageReconstructionDecoder(nn.Module):
    def __init__(self, embed_dim=256, num_layers=4, num_heads=8):
        super().__init__()
        self.grid_h = 16
        self.grid_w = 27
        self.num_patches = self.grid_h * self.grid_w  # 432
        
        self.spatial_queries = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # DC-AE Upsampling (from 16x27 -> 256x432)
        # 4 steps of 2x upsampling = 16x resolution scale
        self.dc_ae_upsampler = nn.Sequential(
            DCAEUpsampleBlock(embed_dim, 128),  # 16x27 -> 32x54
            DCAEUpsampleBlock(128, 64),         # 32x54 -> 64x108
            DCAEUpsampleBlock(64, 32),          # 64x108 -> 128x216
            nn.Conv2d(32, 12, kernel_size=3, padding=1),
            nn.PixelShuffle(2),                 # 128x216 -> 256x432 (Output: 3 Channels for RGB)
            nn.Sigmoid()                        # Assuming pixels are [0, 1] normalized
        )
        
    def forward(self, z, condition_embedding):
        batch_size = z.size(0)
        
        x = self.spatial_queries.expand(batch_size, -1, -1) + self.pos_embed
        memory = z + condition_embedding
        
        for layer in self.layers:
            x = layer(tgt=x, memory=memory)
            
        x = self.final_norm(x)
        
        # Reshape to 2D Spatial Grid: [Batch, 432, 256] -> [Batch, 256, 16, 27]
        x = x.transpose(1, 2).view(batch_size, -1, self.grid_h, self.grid_w)
        
        # Upsample using DC-AE Blocks
        image = self.dc_ae_upsampler(x) # [Batch, 3, 256, 432]
        
        return image

class PoseEstimationHead(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, num_keypoints=17):
        super().__init__()
        self.grid_h = 16
        self.grid_w = 27
        self.num_patches = self.grid_h * self.grid_w 
        self.num_keypoints = num_keypoints
        
        # 1. Spatial Decoder (Transformer)
        self.spatial_queries = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        self.spatial_decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=embed_dim * 4, 
                dropout=0.1, 
                activation="gelu", 
                batch_first=True
            ) for _ in range(2) 
        ])
        
        # 2. Heatmap & AE Tag Predictor (CNN)
        self.heatmap_predictor = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, num_keypoints * 2, kernel_size=1)                             
        )
        
    def forward(self, z, condition_embedding):
        batch_size = z.size(0)
        
        # Unfold compressed 1D latent into rigid 2D layout via spatial queries
        x = self.spatial_queries.expand(batch_size, -1, -1) + self.pos_embed
        
        # Inject the task/ratio/global context into the latent memory
        memory = z + condition_embedding 
        
        for layer in self.spatial_decoder:
            x = layer(tgt=x, memory=memory)
            
        # Reshape to [Batch, Channels, H, W]
        x = x.transpose(1, 2).view(batch_size, -1, self.grid_h, self.grid_w)
        
        # Generate raw 34-channel output
        raw_output = self.heatmap_predictor(x) 
        
        raw_heatmaps = raw_output[:, :self.num_keypoints, :, :]
        ae_tags = raw_output[:, self.num_keypoints:, :, :]
        
        return torch.sigmoid(raw_heatmaps), ae_tags

class ConditionalAutoEncoder(nn.Module):
    def __init__(self, hidden_dim=256, max_image_patches=432, use_precomputed_dino=False):
        super().__init__()
        self.condition_encoder = ConditionEncoder(embed_dim=hidden_dim)
        self.query_generator = RatioSpecificQueryGenerator(max_queries=max_image_patches, embed_dim=hidden_dim)
        self.vision_task_encoder = Encoder(embed_dim=hidden_dim, use_precomputed_dino=use_precomputed_dino)
        
        self.taskhead_humanpose = PoseEstimationHead(embed_dim=hidden_dim, num_keypoints=17)
        self.decoder_imagerecon = ImageReconstructionDecoder(embed_dim=hidden_dim)
    
    def forward(self, task_name: str, target_ratio: float, pixel_values=None, precomputed_features=None):
        # Allow inferring batch_size and device from either pixel_values or precomputed_features
        ref_tensor = precomputed_features if precomputed_features is not None else pixel_values
        batch_size = ref_tensor.size(0)
        device = ref_tensor.device
        
        task_id = torch.tensor([0 if task_name == 'pose' else 1], dtype=torch.long, device=device).expand(batch_size)
        ratio_tensor = torch.tensor([[target_ratio]], dtype=torch.float32, device=device).expand(batch_size, -1)
        
        condition_embedding = self.condition_encoder(task_id, ratio_tensor)
        dynamic_queries = self.query_generator(target_ratio, batch_size)
        
        z, cls_token = self.vision_task_encoder(
            dynamic_queries, condition_embedding, 
            pixel_values=pixel_values, 
            precomputed_features=precomputed_features
        )
        
        enhanced_condition = condition_embedding + cls_token
        
        if task_name == 'pose':
            results = self.taskhead_humanpose(z, enhanced_condition)
        elif task_name == 'recon':
            results = self.decoder_imagerecon(z, enhanced_condition) 
        else:
            raise ValueError(f"Unknown task: {task_name}")
            
        return results, z