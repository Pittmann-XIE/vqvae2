import os
import torch
from torch import nn
from torch.optim import AdamW
import random
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import lpips

from CAE_DINO import ConditionalAutoEncoder
from utils_DINO import COCOBottomUpDataset

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[*] Random seed set to {seed}")

def print_model_size(model):
    """Prints the total and trainable parameter count of the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("="*50)
    print(f"[*] Model Size: {total_params / 1e6:.2f} M total parameters")
    print(f"[*] Trainable : {trainable_params / 1e6:.2f} M parameters")
    print("="*50)

def calculate_pose_loss(preds, gt_heatmaps, gt_grouped_keypoints, alpha=1e-3):
    """
    Calculates the Bottom-Up Pose Loss: Heatmap MSE + AE Push/Pull Loss.
    """
    pred_heatmaps, pred_tags = preds
    
    # 1. Detection Heatmap Loss (MSE)
    heatmap_loss = nn.functional.mse_loss(pred_heatmaps, gt_heatmaps)
    
    # 2. Associative Embedding (Push/Pull) Loss
    batch_size = pred_tags.size(0)
    total_pull_loss = 0.0
    total_push_loss = 0.0
    valid_batches = 0
    
    for b in range(batch_size):
        person_tags = []
        
        # gt_grouped_keypoints: [max_people, 17, 3] -> (x, y, valid)
        for p in range(gt_grouped_keypoints.size(1)):
            tags_for_this_person = []
            for j in range(17):
                x, y, v = gt_grouped_keypoints[b, p, j]
                if v > 0.0:
                    # Cast float coords to integer indices
                    xi, yi = int(x.item()), int(y.item())
                    # Ensure indices are within the 256x432 feature map bounds
                    xi = max(0, min(xi, pred_tags.size(3) - 1))
                    yi = max(0, min(yi, pred_tags.size(2) - 1))
                    
                    # Read the predicted tag value at this joint's exact location
                    tag_val = pred_tags[b, j, yi, xi]
                    tags_for_this_person.append(tag_val)
            
            if len(tags_for_this_person) > 0:
                person_tags.append(torch.stack(tags_for_this_person))
        
        if len(person_tags) == 0:
            continue
            
        valid_batches += 1
        
        # PULL LOSS: Penalize variance of tags belonging to the same person
        pull_loss = 0.0
        person_means = []
        for tags in person_tags:
            mean_tag = tags.mean()
            person_means.append(mean_tag)
            if len(tags) > 1:
                pull_loss += torch.mean((tags - mean_tag)**2)
        
        pull_loss = pull_loss / len(person_tags)
        total_pull_loss += pull_loss
        
        # PUSH LOSS: Penalize if different people have similar mean tags
        push_loss = 0.0
        if len(person_means) > 1:
            person_means = torch.stack(person_means)
            # Pairwise differences between all people
            diffs = person_means.unsqueeze(1) - person_means.unsqueeze(0)
            # Exponential distance penalty (peaks at 1.0 when diff is 0)
            push_dist = torch.exp(- (diffs ** 2))
            
            # Zero out the diagonal (we don't penalize a person against themselves)
            mask = 1.0 - torch.eye(len(person_means), device=push_dist.device)
            push_loss = (push_dist * mask).sum() / (len(person_means) * (len(person_means) - 1))
        
        total_push_loss += push_loss
        
    avg_pull = total_pull_loss / max(1, valid_batches)
    avg_push = total_push_loss / max(1, valid_batches)
    
    # Final combined objective function
    total_pose_loss = heatmap_loss + alpha * (avg_pull + avg_push)
    
    return total_pose_loss

def plot_losses(history, filename="training_loss_plot.png"):
    plt.figure(figsize=(12, 5))
    epochs = range(1, len(history['train_pose']) + 1)
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_pose'], label='Train Pose', marker='o')
    if history['val_pose']:
        plt.plot(epochs, history['val_pose'], label='Val Pose', marker='s')
    plt.title('Pose Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_recon'], label='Train Recon', marker='o')
    if history['val_recon']:
        plt.plot(epochs, history['val_recon'], label='Val Recon', marker='s')
    plt.title('Reconstruction Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close() 

def train_cae_model(model, train_dataloader, val_dataloader=None, args=None, device='cuda'):
    epochs = args.epochs if args else 50
    plot_every = args.plot_every if args else 1
    run_valid = args.valid if args else False
    save_dir = args.save_dir if args else "checkpoints"
    
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    recon_criterion = nn.L1Loss() # Updated to L1 as per requirements
    lpips_criterion = lpips.LPIPS(net='vgg').to(device) # add for image reconstruction
    
    model.to(device)
    
    history = {'train_pose': [], 'train_recon':[], 'val_pose':[], 'val_recon':[]}
    start_epoch = 0
    best_loss = float('inf')
    
    if args and getattr(args, 'resume', None) and os.path.isfile(args.resume):
        print(f"[*] Loading checkpoint '{args.resume}'...")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint.get('best_loss', float('inf'))
        history = checkpoint.get('history', history)
        print(f"[*] Successfully resumed training from Epoch {start_epoch}")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        total_pose_loss, pose_batches = 0.0, 0
        total_recon_loss, recon_batches = 0.0, 0
        
        for batch_idx, batch_data in enumerate(train_dataloader):
            # Gracefully handle either pixel_values OR precomputed_features
            pixel_values = batch_data.get('pixel_values', None)
            precomputed_features = batch_data.get('precomputed_features', None)
            
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            if precomputed_features is not None:
                precomputed_features = precomputed_features.to(device)
                
            gt_heatmaps = batch_data['heatmaps'].to(device)
            gt_grouped_kpts = batch_data['grouped_keypoints'].to(device)
            raw_image_target = batch_data['raw_image'].permute(0, 3, 1, 2).float().to(device) / 255.0
            
            optimizer.zero_grad()
            task_name = random.choice(['pose', 'recon'])
            target_ratio = random.uniform(0.1, 1.0) 
            
            # Pass BOTH kwargs; the model handles whichever is populated
            results, z = model(task_name, target_ratio, pixel_values=pixel_values, precomputed_features=precomputed_features)
            
            # 2. Compute Task-Specific Loss
            if task_name == 'pose':
                loss = calculate_pose_loss(results, gt_heatmaps, gt_grouped_kpts)
                total_pose_loss += loss.item()
                pose_batches += 1
                
            elif task_name == 'recon':
                # 1. Standard L1 Loss (Pixel-level)
                l1_loss = recon_criterion(results, raw_image_target)
                
                # 2. LPIPS Loss (Perceptual-level)
                # LPIPS expects inputs in [-1, 1], so we scale our [0, 1] tensors
                results_scaled = results * 2.0 - 1.0
                target_scaled = raw_image_target * 2.0 - 1.0
                
                # Calculate mean perceptual loss across the batch
                lpips_loss = lpips_criterion(results_scaled, target_scaled).mean()
                
                # Combine losses (You can tweak the 1.0 weight if needed)
                loss = l1_loss + 1.0 * lpips_loss
                
                total_recon_loss += loss.item()
                recon_batches += 1
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if batch_idx % 1 == 0:
                print(f"Epoch[{epoch+1}/{epochs}] Batch {batch_idx} | Task: {task_name.upper():<5} | Ratio: {target_ratio:.2f} | Loss: {loss.item():.4f}")
        
        avg_train_pose = total_pose_loss / max(1, pose_batches)
        avg_train_recon = total_recon_loss / max(1, recon_batches)
        
        history['train_pose'].append(avg_train_pose)
        history['train_recon'].append(avg_train_recon)
        print(f"--- Epoch {epoch+1} Train Complete | Avg Pose Loss: {avg_train_pose:.4f} | Avg Recon Loss: {avg_train_recon:.4f} ---")
        
        avg_val_pose, avg_val_recon = 0.0, 0.0
        
        # ================= VALIDATION LOOP =================

        if run_valid and val_dataloader is not None:
            model.eval()
            val_pose_loss, val_recon_loss = 0.0, 0.0
            
            with torch.no_grad():
                for batch_data in val_dataloader:
                    pixel_values = batch_data.get('pixel_values', None)
                    precomputed_features = batch_data.get('precomputed_features', None)
                    
                    if pixel_values is not None:
                        pixel_values = pixel_values.to(device)
                    if precomputed_features is not None:
                        precomputed_features = precomputed_features.to(device)

                    gt_heatmaps = batch_data['heatmaps'].to(device)
                    gt_grouped_kpts = batch_data['grouped_keypoints'].to(device)
                    raw_image_target = batch_data['raw_image'].permute(0, 3, 1, 2).float().to(device) / 255.0
                    
                    results_pose, _ = model('pose', 1.0, pixel_values=pixel_values, precomputed_features=precomputed_features)
                    val_pose_loss += calculate_pose_loss(results_pose, gt_heatmaps, gt_grouped_kpts).item()
                    
                    results_recon, _ = model('recon', 1.0, pixel_values=pixel_values, precomputed_features=precomputed_features)
                    # Compute combined validation recon loss
                    val_l1 = recon_criterion(results_recon, raw_image_target)
                    val_res_scaled = results_recon * 2.0 - 1.0
                    val_tgt_scaled = raw_image_target * 2.0 - 1.0
                    val_lpips = lpips_criterion(val_res_scaled, val_tgt_scaled).mean()
                    
                    val_recon_loss += (val_l1 + val_lpips).item()
            
            avg_val_pose = val_pose_loss / len(val_dataloader)
            avg_val_recon = val_recon_loss / len(val_dataloader)
            
            history['val_pose'].append(avg_val_pose)
            history['val_recon'].append(avg_val_recon)
            print(f"--- Epoch {epoch+1} Val Complete   | Avg Pose Loss: {avg_val_pose:.4f} | Avg Recon Loss: {avg_val_recon:.4f} ---")

        # ================= CHECKPOINTING & PLOTTING =================
        current_combined_loss = (avg_val_pose + avg_val_recon) if run_valid else (avg_train_pose + avg_train_recon)
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': current_combined_loss,
            'best_loss': best_loss,
            'history': history
        }

        if current_combined_loss < best_loss:
            best_loss = current_combined_loss
            best_ckpt_path = os.path.join(save_dir, "best_checkpoint.pth")
            torch.save(checkpoint_state, best_ckpt_path)
            print(f"[*] New best model saved to {best_ckpt_path} (Combined Loss: {best_loss:.4f})")

        if (epoch + 1) % plot_every == 0:
            plot_losses(history, filename="training_loss_plot_DINO.png")
            print(f"[*] Updated plot at training_loss_plot_DINO.png")
            
            periodic_ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint_state, periodic_ckpt_path)
            print(f"[*] Checkpoint saved to {periodic_ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConditionalAutoEncoder")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--valid", action="store_true", help="Enable validation loop")
    parser.add_argument("--plot-every", type=int, default=20, help="Update loss plot & save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=10, help="Random seed for reproducibility")
    parser.add_argument("--save-dir", type=str, default="checkpoints_DINO", help="Directory to save weights")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume training from")
    
    # NEW FLAG: cleanly toggle between precomputed features and raw pixels
    parser.add_argument("--use-precomputed", action="store_true", help="Use precomputed DINO features instead of raw pixels")
    
    args = parser.parse_args()

    set_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Initialize the Model (Dynamically respects the CLI flag)
    cae_model = ConditionalAutoEncoder(hidden_dim=256, max_image_patches=432, use_precomputed_dino=args.use_precomputed)
    print_model_size(cae_model)
    
    # 2. Setup the Data Paths
    DATA_DIR = os.path.expanduser("~/ws_ros2humble-main_lab/vqvae/data_all")
    
    # Conditionally set feature directories based on the flag
    train_features_dir = os.path.join(DATA_DIR, "dino_features_train2017") if args.use_precomputed else None
    val_features_dir = os.path.join(DATA_DIR, "dino_features_val2017") if args.use_precomputed else None
    
    # 3. Instantiate the Bottom-Up Datasets
    train_dataset = COCOBottomUpDataset(
        data_dir=DATA_DIR,
        ann_file="annotations/person_keypoints_train2017.json", 
        img_dir="coco/train2017",
        features_dir=train_features_dir
    )
    
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    val_loader = None
    if args.valid:
        val_dataset = COCOBottomUpDataset(
            data_dir=DATA_DIR,
            ann_file="annotations/person_keypoints_val2017.json", 
            img_dir="coco/val2017",
            features_dir=val_features_dir
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g
        )
    
    # 4. Start Training!
    train_cae_model(cae_model, train_loader, val_dataloader=val_loader, args=args, device=device)