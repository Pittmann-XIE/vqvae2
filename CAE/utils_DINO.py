import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import torchvision.transforms as T
import matplotlib.pyplot as plt
import random

def draw_gaussian(heatmap, center, sigma=2):
    """
    Draws a 2D Gaussian distribution on a specific heatmap channel at the given center coordinate.
    """
    x, y = int(center[0]), int(center[1])
    h, w = heatmap.shape
    
    tmp_size = sigma * 3
    ul = [int(x - tmp_size), int(y - tmp_size)]
    br = [int(x + tmp_size + 1), int(y + tmp_size + 1)]
    
    # Check if the Gaussian is completely out of bounds
    if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0:
        return heatmap
        
    size = 2 * tmp_size + 1
    x_grid = np.arange(0, size, 1, np.float32)
    y_grid = x_grid[:, np.newaxis]
    x0, y0 = size // 2, size // 2
    
    # Generate the Gaussian
    g = np.exp(- ((x_grid - x0) ** 2 + (y_grid - y0) ** 2) / (2 * sigma ** 2))
    
    # Usable area of the Gaussian and the heatmap
    g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], h) - ul[1]
    img_x = max(0, ul[0]), min(br[0], w)
    img_y = max(0, ul[1]), min(br[1], h)
    
    # Apply maximum to prevent overlapping joints from overwriting each other destructively
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    )
    return heatmap

class COCOBottomUpDataset(Dataset):
    def __init__(self, data_dir, ann_file, img_dir, target_h=242, target_w=424, pad_h=256, pad_w=432, max_people=30, return_orig_for_vis=False, features_dir=None):
        self.img_dir = os.path.join(data_dir, img_dir)
        self.target_h = target_h
        self.target_w = target_w
        self.pad_h = pad_h
        self.pad_w = pad_w
        self.max_people = max_people
        self.return_orig_for_vis = return_orig_for_vis
        
        ann_path = os.path.join(data_dir, ann_file)
        print(f"Loading COCO annotations from {ann_path}...")
        self.coco = COCO(ann_path)
        
        # 1. Identify valid FULL IMAGES (containing at least one person with keypoints)
        catIds = self.coco.getCatIds(catNms=['person'])
        imgIds = self.coco.getImgIds(catIds=catIds)
        
        self.valid_img_ids = []
        for img_id in imgIds:
            ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=catIds)
            anns = self.coco.loadAnns(ann_ids)
            # Ensure at least one person in the image actually has labeled keypoints
            if any(ann.get('num_keypoints', 0) > 0 for ann in anns):
                self.valid_img_ids.append(img_id)
                
        print(f"Found {len(self.valid_img_ids)} valid images for multi-person bottom-up training.")

        # Replace CLIP Processor with standard ImageNet normalization (used by DINOv3)
        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.features_dir = features_dir
        if self.features_dir:
            print(f"[*] Dataset initialized to load precomputed features from {self.features_dir}")

    def __len__(self):
        return len(self.valid_img_ids)

    def __getitem__(self, idx):
        img_id = self.valid_img_ids[idx]
        
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w, _ = image.shape
        
        # 1. Letterbox Scaling: Fit longest edge inside 242x424
        scale = min(self.target_w / orig_w, self.target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        resized_img = cv2.resize(image, (new_w, new_h))
        
        # 2. Centered Rectangular Padding: Calculate offsets
        pad_left = (self.pad_w - new_w) // 2
        pad_top = (self.pad_h - new_h) // 2
        
        # Paste onto the center of a 256x432 neutral gray canvas
        canvas = np.full((self.pad_h, self.pad_w, 3), 114, dtype=np.uint8)
        canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w, :] = resized_img
        
        # 3. Retrieve ALL multi-person annotations for this scene
        ann_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.coco.getCatIds(catNms=['person']))
        anns = self.coco.loadAnns(ann_ids)
        
        # Initialize output tensors
        heatmaps = np.zeros((17, self.pad_h, self.pad_w), dtype=np.float32)
        # [max_people, 17, 3 (x, y, valid)] - Used for AE push/pull loss
        grouped_keypoints = np.zeros((self.max_people, 17, 3), dtype=np.float32) 
        
        person_count = 0
        raw_keypoints_list = [] # Stored strictly for unit-testing/visualization

        for ann in anns:
            if ann.get('num_keypoints', 0) == 0:
                continue
            if person_count >= self.max_people:
                break # Cap max people to keep tensor sizes uniform for DataLoader
                
            raw_kpts = np.array(ann['keypoints']).reshape(17, 3)
            raw_keypoints_list.append(raw_kpts) 
            
            for i in range(17):
                kx, ky, kv = raw_kpts[i]
                
                if kv > 0:
                    # Apply the letterbox scale AND the centering offsets
                    nx = (kx * scale) + pad_left
                    ny = (ky * scale) + pad_top
                    
                    if 0 <= nx < self.pad_w and 0 <= ny < self.pad_h:
                        # Draw Gaussian probability on Detection Heatmap
                        heatmaps[i] = draw_gaussian(heatmaps[i], (nx, ny), sigma=2)
                        
                        # Save exact coordinate and person_id for Associative Embedding Tags
                        grouped_keypoints[person_count, i, 0] = nx
                        grouped_keypoints[person_count, i, 1] = ny
                        grouped_keypoints[person_count, i, 2] = 1.0 

            person_count += 1

        # 4. Apply DINOv3 Normalization
        pixel_values = self.normalize(canvas) 
        
        output = {
            'img_id': img_id,                      
            'pixel_values': pixel_values,
            'heatmaps': torch.tensor(heatmaps, dtype=torch.float32),
            'grouped_keypoints': torch.tensor(grouped_keypoints, dtype=torch.float32),
            'raw_image': canvas # Clean RGB target for the Image Reconstruction Decoder
        }

        if self.features_dir:
            feature_path = os.path.join(self.features_dir, f"{img_id}.pt")
            # Load the float16 tensor, convert back to float32 for the model
            output['precomputed_features'] = torch.load(feature_path, weights_only=True).float()
            # If using precomputed features, we don't strictly need pixel_values anymore
            # but keeping it doesn't hurt. You can delete it to save CPU-GPU transfer bandwidth.
            del output['pixel_values']
        
        # Bypasses collation issues when fetching direct samples for visualization
        if self.return_orig_for_vis:
            output['orig_image'] = image
            output['orig_keypoints'] = raw_keypoints_list
            
        return output

# ==========================================
# USAGE EXAMPLE & UNIT TEST
# ==========================================
if __name__ == "__main__":
    DATA_DIR = os.path.expanduser("~/ws_ros2humble-main_lab/vqvae/data")
    
    # We set return_orig_for_vis=True to get original shapes. 
    # NOTE: Do NOT use a DataLoader with batch_size > 1 when return_orig_for_vis=True 
    # because standard collate_fn will crash on variable sized original images.
    val_dataset = COCOBottomUpDataset(
        data_dir=DATA_DIR,
        ann_file="annotations/person_keypoints_val2017.json",
        img_dir="coco/val2017",
        return_orig_for_vis=True
    )
    
    # Select 4 random unique indices from the dataset
    random_indices = np.random.choice(len(val_dataset), size=4, replace=False)
    print(f"Visualizing random dataset indices: {random_indices}")
    
    # Create 2 Rows (Original vs Processed), 4 Columns (Samples)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for i, dataset_idx in enumerate(random_indices):
        # Fetch the random sample directly from dataset
        sample = val_dataset[dataset_idx]
        
        orig_img = sample['orig_image']
        orig_kpts_list = sample['orig_keypoints']
        
        padded_img = sample['raw_image']
        grouped_kpts = sample['grouped_keypoints'].numpy()
        
        # --- ROW 1: Plot Original Images & Keypoints ---
        axes[0, i].imshow(orig_img)
        axes[0, i].set_title(f"Original Scene (Idx {dataset_idx})")
        axes[0, i].axis('off')
        
        for person_kpts in orig_kpts_list:
            for j in range(17):
                x, y, v = person_kpts[j]
                if v > 0:
                    axes[0, i].scatter(x, y, c='red', s=15, marker='o')
                    
        # --- ROW 2: Plot Processed (Padded) Images & Keypoints ---
        axes[1, i].imshow(padded_img)
        axes[1, i].set_title(f"256x432 Padded Target {i+1}")
        axes[1, i].axis('off')
        
        for p_idx in range(grouped_kpts.shape[0]):
            for j in range(17):
                x, y, v = grouped_kpts[p_idx, j]
                if v > 0.0:
                    axes[1, i].scatter(x, y, c='lime', s=15, marker='o')
                    
    plt.tight_layout()
    save_path = "debug_bottomup_padding.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved multi-person visualization grid to: {save_path}")