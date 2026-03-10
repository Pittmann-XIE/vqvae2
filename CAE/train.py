# import os
# import torch
# from torch import nn
# from torch.optim import AdamW
# import random
# import numpy as np
# import argparse
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader

# from CAE import ConditionalAutoEncoder
# from utils import COCOTopDownDataset

# def set_seed(seed):
#     """Sets the seed for reproducibility."""
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
#     print(f"[*] Random seed set to {seed}")

# def print_model_size(model):
#     """Prints the total and trainable parameter count of the model."""
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print("="*50)
#     print(f"[*] Model Size: {total_params / 1e6:.2f} M total parameters")
#     print(f"[*] Trainable : {trainable_params / 1e6:.2f} M parameters")
#     print("="*50)

# def calculate_pose_loss(pred_keypoints, gt_keypoints):
#     """
#     Calculates the loss for 17 YOLO-style keypoints.
#     pred_keypoints:[Batch, 17, 3] -> (x, y, confidence)
#     gt_keypoints:[Batch, 17, 3] -> (x, y, visibility_flag)
#     """
#     pred_coords = pred_keypoints[..., :2]
#     pred_conf = pred_keypoints[..., 2]
    
#     gt_coords = gt_keypoints[..., :2]
#     gt_vis = gt_keypoints[..., 2] 
    
#     coord_loss = nn.functional.mse_loss(pred_coords, gt_coords, reduction='none')
#     coord_loss = (coord_loss.sum(dim=-1) * gt_vis).mean()
    
#     conf_loss = nn.functional.binary_cross_entropy(pred_conf, gt_vis)
    
#     return (2.0 * coord_loss) + conf_loss

# def plot_losses(history, filename="training_loss_plot.png"):
#     """
#     Plots the training (and optional validation) losses into a single image.
#     """
#     plt.figure(figsize=(12, 5))
#     epochs = range(1, len(history['train_pose']) + 1)
    
#     # 1. Pose Loss Subplot
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs, history['train_pose'], label='Train Pose', marker='o')
#     if history['val_pose']:
#         plt.plot(epochs, history['val_pose'], label='Val Pose', marker='s')
#     plt.title('Pose Loss Over Time')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.grid(True)
#     plt.legend()
    
#     # 2. Reconstruction Loss Subplot
#     plt.subplot(1, 2, 2)
#     plt.plot(epochs, history['train_recon'], label='Train Recon', marker='o')
#     if history['val_recon']:
#         plt.plot(epochs, history['val_recon'], label='Val Recon', marker='s')
#     plt.title('Reconstruction Loss Over Time')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.grid(True)
#     plt.legend()
    
#     plt.tight_layout()
#     plt.savefig(filename)
#     plt.close() # Close to free up memory

# def train_cae_model(model, train_dataloader, val_dataloader=None, args=None, device='cuda'):
#     epochs = args.epochs if args else 50
#     plot_every = args.plot_every if args else 1
#     run_valid = args.valid if args else False
    
#     optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate)
#     recon_criterion = nn.MSELoss()
    
#     model.to(device)
    
#     # Tracking dictionary for plotting
#     history = {
#         'train_pose': [], 'train_recon':[],
#         'val_pose':[], 'val_recon':[]
#     }
    
#     for epoch in range(epochs):
#         model.train()
#         total_pose_loss, pose_batches = 0.0, 0
#         total_recon_loss, recon_batches = 0.0, 0
        
#         for batch_idx, batch_data in enumerate(train_dataloader):
#             pixel_values = batch_data['pixel_values'].to(device)
#             gt_keypoints = batch_data['keypoints'].to(device)
            
#             optimizer.zero_grad()
            
#             task_name = random.choice(['pose', 'recon'])
#             target_ratio = random.uniform(0.1, 1.0) 
            
#             results, z = model(pixel_values, task_name, target_ratio)
            
#             if task_name == 'pose':
#                 loss = calculate_pose_loss(results, gt_keypoints)
#                 total_pose_loss += loss.item()
#                 pose_batches += 1
                
#             elif task_name == 'recon':
#                 loss = recon_criterion(results, pixel_values)
#                 total_recon_loss += loss.item()
#                 recon_batches += 1
            
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#             optimizer.step()
            
#             if batch_idx % 50 == 0:
#                 print(f"Epoch [{epoch+1}/{epochs}] Batch {batch_idx} | Task: {task_name.upper():<5} | Ratio: {target_ratio:.2f} | Loss: {loss.item():.4f}")
        
#         # Calculate accurate averages based on how many times each task was picked
#         avg_train_pose = total_pose_loss / max(1, pose_batches)
#         avg_train_recon = total_recon_loss / max(1, recon_batches)
        
#         history['train_pose'].append(avg_train_pose)
#         history['train_recon'].append(avg_train_recon)
        
#         print(f"--- Epoch {epoch+1} Train Complete | Avg Pose Loss: {avg_train_pose:.4f} | Avg Recon Loss: {avg_train_recon:.4f} ---")
        
#         # ================= VALIDATION LOOP =================
#         if run_valid and val_dataloader is not None:
#             model.eval()
#             val_pose_loss, val_recon_loss = 0.0, 0.0
            
#             with torch.no_grad():
#                 for batch_data in val_dataloader:
#                     pixel_values = batch_data['pixel_values'].to(device)
#                     gt_keypoints = batch_data['keypoints'].to(device)
                    
#                     # Eval Pose (Evaluate at full target ratio for consistency)
#                     results_pose, _ = model(pixel_values, 'pose', 1.0)
#                     val_pose_loss += calculate_pose_loss(results_pose, gt_keypoints).item()
                    
#                     # Eval Recon
#                     results_recon, _ = model(pixel_values, 'recon', 1.0)
#                     val_recon_loss += recon_criterion(results_recon, pixel_values).item()
            
#             avg_val_pose = val_pose_loss / len(val_dataloader)
#             avg_val_recon = val_recon_loss / len(val_dataloader)
            
#             history['val_pose'].append(avg_val_pose)
#             history['val_recon'].append(avg_val_recon)
            
#             print(f"--- Epoch {epoch+1} Val Complete   | Avg Pose Loss: {avg_val_pose:.4f} | Avg Recon Loss: {avg_val_recon:.4f} ---")

#         # ================= PLOTTING =================
#         if (epoch + 1) % plot_every == 0:
#             plot_losses(history, filename="training_loss_plot.png")
#             print(f"[*] Updated plot at training_loss_plot.png")


# if __name__ == "__main__":
#     # 0. Argument Parser
#     parser = argparse.ArgumentParser(description="Train ConditionalAutoEncoder")
#     parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
#     parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
#     parser.add_argument("--valid", action="store_true", help="Enable validation loop")
#     parser.add_argument("--plot-every", type=int, default=1, help="Update loss plot every N epochs")
#     parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
#     parser.add_argument("--learning_rate", type=int, default=1e-4, help="Learning rate")

#     args = parser.parse_args()

#     # Apply Random Seed
#     set_seed(args.seed)

#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     # 1. Initialize the Model
#     cae_model = ConditionalAutoEncoder(hidden_dim=256, max_image_patches=196)
    
#     # 2. Print Model Size
#     print_model_size(cae_model)
    
#     # 3. Setup the Data Paths
#     DATA_DIR = os.path.expanduser("~/ws_ros2humble-main_lab/vqvae/data")
    
#     # 4. Instantiate the Datasets (Using train2017 for training, val2017 for validation)
#     train_dataset = COCOTopDownDataset(
#         data_dir=DATA_DIR,
#         ann_file="annotations/person_keypoints_val2017.json", 
#         img_dir="coco/val2017",
#         target_size=224
#     )
    
#     # Create generators/workers seed ensuring correct distributed seeding behavior 
#     g = torch.Generator()
#     g.manual_seed(args.seed)
    
#     def seed_worker(worker_id):
#         worker_seed = torch.initial_seed() % 2**32
#         np.random.seed(worker_seed)
#         random.seed(worker_seed)
        
#     train_loader = DataLoader(
#         train_dataset, 
#         batch_size=args.batch_size, 
#         shuffle=True, 
#         num_workers=4,
#         pin_memory=True,
#         worker_init_fn=seed_worker,
#         generator=g
#     )
    
#     val_loader = None
#     if args.valid:
#         val_dataset = COCOTopDownDataset(
#             data_dir=DATA_DIR,
#             ann_file="annotations/person_keypoints_val2017.json", 
#             img_dir="coco/val2017",
#             target_size=224
#         )
#         val_loader = DataLoader(
#             val_dataset,
#             batch_size=args.batch_size,
#             shuffle=False,  # No need to shuffle validation
#             num_workers=4,
#             pin_memory=True,
#             worker_init_fn=seed_worker,
#             generator=g
#         )
    
#     # 5. Start Training!
#     train_cae_model(cae_model, train_loader, val_dataloader=val_loader, args=args, device=device)



##
import os
import torch
from torch import nn
from torch.optim import AdamW
import random
import numpy as np
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from CAE import ConditionalAutoEncoder
from utils import COCOTopDownDataset

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
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

def calculate_pose_loss(pred_keypoints, gt_keypoints):
    """
    Calculates the loss for 17 YOLO-style keypoints.
    pred_keypoints:[Batch, 17, 3] -> (x, y, confidence)
    gt_keypoints:[Batch, 17, 3] -> (x, y, visibility_flag)
    """
    pred_coords = pred_keypoints[..., :2]
    pred_conf = pred_keypoints[..., 2]
    
    gt_coords = gt_keypoints[..., :2]
    gt_vis = gt_keypoints[..., 2] 
    
    coord_loss = nn.functional.mse_loss(pred_coords, gt_coords, reduction='none')
    coord_loss = (coord_loss.sum(dim=-1) * gt_vis).mean()
    
    conf_loss = nn.functional.binary_cross_entropy(pred_conf, gt_vis)
    
    return (2.0 * coord_loss) + conf_loss

def plot_losses(history, filename="training_loss_plot.png"):
    """
    Plots the training (and optional validation) losses into a single image.
    """
    plt.figure(figsize=(12, 5))
    epochs = range(1, len(history['train_pose']) + 1)
    
    # 1. Pose Loss Subplot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_pose'], label='Train Pose', marker='o')
    if history['val_pose']:
        plt.plot(epochs, history['val_pose'], label='Val Pose', marker='s')
    plt.title('Pose Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # 2. Reconstruction Loss Subplot
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
    plt.close() # Close to free up memory

def train_cae_model(model, train_dataloader, val_dataloader=None, args=None, device='cuda'):
    epochs = args.epochs if args else 50
    plot_every = args.plot_every if args else 1
    run_valid = args.valid if args else False
    save_dir = args.save_dir if args else "checkpoints"
    
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    recon_criterion = nn.MSELoss()
    
    model.to(device)
    
    # Tracking dictionaries and states
    history = {
        'train_pose': [], 'train_recon':[],
        'val_pose':[], 'val_recon':[]
    }
    start_epoch = 0
    best_loss = float('inf')
    
    # ================= RESUME CHECKPOINT =================
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
            pixel_values = batch_data['pixel_values'].to(device)
            gt_keypoints = batch_data['keypoints'].to(device)
            
            optimizer.zero_grad()
            
            task_name = random.choice(['pose', 'recon'])
            target_ratio = random.uniform(0.1, 1.0) 
            
            results, z = model(pixel_values, task_name, target_ratio)
            
            if task_name == 'pose':
                loss = calculate_pose_loss(results, gt_keypoints)
                total_pose_loss += loss.item()
                pose_batches += 1
                
            elif task_name == 'recon':
                loss = recon_criterion(results, pixel_values)
                total_recon_loss += loss.item()
                recon_batches += 1
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if batch_idx % 50 == 0:
                print(f"Epoch[{epoch+1}/{epochs}] Batch {batch_idx} | Task: {task_name.upper():<5} | Ratio: {target_ratio:.2f} | Loss: {loss.item():.4f}")
        
        # Calculate accurate averages based on how many times each task was picked
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
                    pixel_values = batch_data['pixel_values'].to(device)
                    gt_keypoints = batch_data['keypoints'].to(device)
                    
                    # Eval Pose (Evaluate at full target ratio for consistency)
                    results_pose, _ = model(pixel_values, 'pose', 1.0)
                    val_pose_loss += calculate_pose_loss(results_pose, gt_keypoints).item()
                    
                    # Eval Recon
                    results_recon, _ = model(pixel_values, 'recon', 1.0)
                    val_recon_loss += recon_criterion(results_recon, pixel_values).item()
            
            avg_val_pose = val_pose_loss / len(val_dataloader)
            avg_val_recon = val_recon_loss / len(val_dataloader)
            
            history['val_pose'].append(avg_val_pose)
            history['val_recon'].append(avg_val_recon)
            
            print(f"--- Epoch {epoch+1} Val Complete   | Avg Pose Loss: {avg_val_pose:.4f} | Avg Recon Loss: {avg_val_recon:.4f} ---")

        # ================= CHECKPOINTING & PLOTTING =================
        
        # Determine the metric to track for "best" model
        current_combined_loss = (avg_val_pose + avg_val_recon) if run_valid else (avg_train_pose + avg_train_recon)
        
        checkpoint_state = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': current_combined_loss,
            'best_loss': best_loss,
            'history': history
        }

        # 1. Save Best Checkpoint
        if current_combined_loss < best_loss:
            best_loss = current_combined_loss
            best_ckpt_path = os.path.join(save_dir, "best_checkpoint.pth")
            torch.save(checkpoint_state, best_ckpt_path)
            print(f"[*] New best model saved to {best_ckpt_path} (Combined Loss: {best_loss:.4f})")

        # 2. Plot and Save Periodic Checkpoint
        if (epoch + 1) % plot_every == 0:
            # Update Plot
            plot_losses(history, filename="training_loss_plot.png")
            print(f"[*] Updated plot at training_loss_plot.png")
            
            # Save periodic checkpoint for continuing
            periodic_ckpt_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint_state, periodic_ckpt_path)
            print(f"[*] Checkpoint saved to {periodic_ckpt_path}")


if __name__ == "__main__":
    # 0. Argument Parser
    parser = argparse.ArgumentParser(description="Train ConditionalAutoEncoder")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--valid", action="store_true", help="Enable validation loop")
    parser.add_argument("--plot-every", type=int, default=1, help="Update loss plot & save checkpoint every N epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="Directory to save weights")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    # Apply Random Seed
    set_seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Initialize the Model
    cae_model = ConditionalAutoEncoder(hidden_dim=256, max_image_patches=196)
    
    # 2. Print Model Size
    print_model_size(cae_model)
    
    # 3. Setup the Data Paths
    DATA_DIR = os.path.expanduser("~/ws_ros2humble-main_lab/vqvae/data_all")
    
    # 4. Instantiate the Datasets (Using train2017 for training, val2017 for validation)
    train_dataset = COCOTopDownDataset(
        data_dir=DATA_DIR,
        ann_file="annotations/person_keypoints_train2017.json", 
        img_dir="coco/train2017",
        target_size=224
    )
    
    # Create generators/workers seed ensuring correct distributed seeding behavior 
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
        val_dataset = COCOTopDownDataset(
            data_dir=DATA_DIR,
            ann_file="annotations/person_keypoints_val2017.json", 
            img_dir="coco/val2017",
            target_size=224
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,  # No need to shuffle validation
            num_workers=4,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g
        )
    
    # 5. Start Training!
    train_cae_model(cae_model, train_loader, val_dataloader=val_loader, args=args, device=device)