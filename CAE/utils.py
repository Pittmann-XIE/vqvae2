import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor
import matplotlib.pyplot as plt

class COCOTopDownDataset(Dataset):
    def __init__(self, data_dir, ann_file, img_dir, target_size=224):
        self.target_size = target_size
        self.img_dir = os.path.join(data_dir, img_dir)
        
        ann_path = os.path.join(data_dir, ann_file)
        print(f"Loading COCO annotations from {ann_path}...")
        self.coco = COCO(ann_path)
        
        self.valid_anns = []
        for ann_id in self.coco.getAnnIds(catIds=self.coco.getCatIds(catNms=['person'])):
            ann = self.coco.loadAnns(ann_id)[0]
            if ann['num_keypoints'] > 0 and 'bbox' in ann:
                self.valid_anns.append(ann)
                
        print(f"Found {len(self.valid_anns)} valid single-person training crops.")

        self.processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch16",
            use_safetensors=True,
            do_resize=False,
            do_center_crop=False,
            do_rescale=True,
            do_normalize=True
        )

    def __len__(self):
        return len(self.valid_anns)

    def __getitem__(self, idx):
        ann = self.valid_anns[idx]
        
        img_info = self.coco.loadImgs(ann['image_id'])[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = image.shape
        
        # 1. Get Bounding Box
        x, y, w, h = [int(v) for v in ann['bbox']]
        
        # 2. Determine a Random Square Crop Size (S)
        # Minimum size needed to fit the person
        min_s = max(w, h)
        
        # Maximum possible square we can extract from this image
        max_s = min(img_w, img_h)
        min_s = min(min_s, max_s) # Safety catch for bad COCO annotations
        
        # To prevent blur, aim for a crop size of AT LEAST 224 (if the image allows it)
        # We also randomly expand the box by 1.2x to 2.0x to include environmental context
        ideal_min_s = max(int(min_s * np.random.uniform(1.2, 2.0)), self.target_size)
        lower_bound = min(ideal_min_s, max_s)
        
        # Pick a random square size between our lower bound and the max image size
        if lower_bound < max_s:
            S = np.random.randint(lower_bound, max_s + 1)
        else:
            S = lower_bound
            
        # 3. Determine Crop Origin (X1, Y1)
        # The crop must fully contain the bbox and stay within image boundaries
        x_min = max(0, x + w - S)
        x_max = min(x, img_w - S)
        y_min = max(0, y + h - S)
        y_max = min(y, img_h - S)
        
        # Safe random selection of the top-left corner of the crop
        X1 = np.random.randint(x_min, x_max + 1) if x_min <= x_max else max(0, x + w - S)
        Y1 = np.random.randint(y_min, y_max + 1) if y_min <= y_max else max(0, y + h - S)
        
        # 4. Crop and Resize
        cropped_img = image[Y1:Y1+S, X1:X1+S]
        
        # Failsafe padding in case bbox was out of image bounds (rare COCO edge case)
        ch, cw, _ = cropped_img.shape
        if ch != S or cw != S:
            cropped_img = cv2.copyMakeBorder(cropped_img, 0, max(0, S - ch), 0, max(0, S - cw), cv2.BORDER_CONSTANT, value=(114, 114, 114))
            
        # Because the crop is a perfect square (S x S), resizing preserves aspect ratio natively!
        resized_img = cv2.resize(cropped_img, (self.target_size, self.target_size))
        scale = self.target_size / S
        
        # 5. Transform Keypoints
        raw_kpts = np.array(ann['keypoints']).reshape(17, 3)
        final_kpts = np.zeros((17, 3), dtype=np.float32)
        
        for i in range(17):
            kx, ky, kv = raw_kpts[i]
            
            if kv > 0:
                # Shift relative to the new random crop window
                new_x = (kx - X1) * scale
                new_y = (ky - Y1) * scale
                
                # Normalize to [0.0, 1.0]
                final_kpts[i][0] = new_x / self.target_size
                final_kpts[i][1] = new_y / self.target_size
                final_kpts[i][2] = 1.0 
            else:
                final_kpts[i] = [0.0, 0.0, 0.0]

        # 6. Apply CLIP Normalization
        inputs = self.processor(images=resized_img, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0) 
        
        return {
            'pixel_values': pixel_values,
            'keypoints': torch.tensor(final_kpts, dtype=torch.float32),
            'raw_image': resized_img # Send the clean RGB image for visualization/reconstruction target
        }

# ==========================================
# USAGE EXAMPLE & UNIT TEST
# ==========================================
if __name__ == "__main__":
    DATA_DIR = os.path.expanduser("~/ws_ros2humble-main_lab/vqvae/data")
    
    val_dataset = COCOTopDownDataset(
        data_dir=DATA_DIR,
        ann_file="annotations/person_keypoints_val2017.json",
        img_dir="coco/val2017",
        target_size=224
    )
    
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=0)
    
    for batch in val_loader:
        keypoints = batch['keypoints']
        raw_images = batch['raw_image'] 
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for i in range(4):
            img = raw_images[i].numpy()
            kpts = keypoints[i].numpy()
            
            axes[i].imshow(img)
            axes[i].set_title(f"Random Square Crop {i+1}")
            axes[i].axis('off')
            
            for j in range(17):
                x_norm, y_norm, v = kpts[j]
                if v > 0.0: 
                    x_pixel = x_norm * 224
                    y_pixel = y_norm * 224
                    axes[i].scatter(x_pixel, y_pixel, c='red', s=20, marker='o')
                    axes[i].text(x_pixel + 2, y_pixel + 2, str(j), color='lime', fontsize=8)
                    
        plt.tight_layout()
        save_path = "debug_keypoints_randomcrop.png"
        plt.savefig(save_path, dpi=150)
        print(f"\nSaved visualization grid to: {save_path}")
        break