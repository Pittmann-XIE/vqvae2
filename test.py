import os
import time
import struct
import numpy as np
import torch
import math
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# Assume networks.py exists
from networks import VQVAE

# ==========================================
# CONFIGURATION
# ==========================================
MODE = 'decompress'  # Options: 'compress' or 'decompress'

# Paths
INPUT_DIR = '../datasets/dummy/valid'       
COMPRESSED_DIR = 'transformed_images/A/compressed' 
RECON_DIR = 'transformed_images/A/recon'   
# Folder to save the resized/padded originals (Ground Truth for PSNR)
ORIG_SAVE_DIR = 'transformed_images/A/original' 

# Model Config
MODEL_NAME = 'A'
IMG_SIZE = 256
ENCODE_DEVICE = torch.device('cpu')   
DECODE_DEVICE = torch.device('cuda')  

# Normalization
MEAN_VALS = [0.485, 0.456, 0.406]
STD_VALS = [0.229, 0.224, 0.225]

# ==========================================
# UTILS
# ==========================================

def get_model(model_name, target_device):
    if model_name == 'A':
        model_path = '/home/xie/vqvae2/checkpoints/COCO/0/checkpoint/vqvae_020.pt'
        model = VQVAE(first_stride=4, second_stride=2)
    elif model_name == 'D':
        model_path = '/home/aisinai/work/VQ-VAE2/20200422/vq_vae/CheXpert/embed1/checkpoint/vqvae_040.pt'
        model = VQVAE(first_stride=4, second_stride=2, embed_dim=1)
    else:
        raise ValueError(f"Model {model_name} path not defined.")

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=target_device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    
    model = model.to(target_device).eval()
    return model

def preprocess_image(im_pth, target_size):
    im = Image.open(im_pth).convert('RGB')
    old_size = im.size
    ratio = float(target_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    im = im.resize(new_size, Image.Resampling.LANCZOS)

    new_im = Image.new("RGB", (target_size, target_size))
    new_im.paste(im, ((target_size - new_size[0]) // 2,
                      (target_size - new_size[1]) // 2))
    return new_im

def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def measure_execution(device, func, *args, **kwargs):
    """
    Accurately measures execution time.
    Synchronizes CUDA before and after the function call.
    """
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    
    start_t = time.perf_counter()
    result = func(*args, **kwargs)
    
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        
    end_t = time.perf_counter()
    return result, end_t - start_t

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_VALS, std=STD_VALS)
])

# ==========================================
# MAIN EXECUTION
# ==========================================

def run_process():
    os.makedirs(COMPRESSED_DIR, exist_ok=True)
    os.makedirs(RECON_DIR, exist_ok=True)
    os.makedirs(ORIG_SAVE_DIR, exist_ok=True)

    active_device = ENCODE_DEVICE if MODE == 'compress' else DECODE_DEVICE
    model = get_model(MODEL_NAME, active_device)

    # 1. Collect Files
    if MODE == 'compress':
        files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        files = [f for f in os.listdir(COMPRESSED_DIR) if f.endswith('.bin')]
    
    files.sort()
    print(f"--- Starting {MODE.upper()} on {len(files)} files using {active_device} ---")

    # ------------------------------------------------------
    # PHASE 1: PROCESS & TIME (No Metric Calculation here)
    # ------------------------------------------------------
    timings = []
    
    for idx, filename in enumerate(files):
        file_base = os.path.splitext(filename)[0]

        if MODE == 'compress':
            # Load & Preprocess
            input_path = os.path.join(INPUT_DIR, filename)
            pil_image = preprocess_image(input_path, IMG_SIZE)
            
            # Save reference copy for later PSNR calculation
            orig_save_path = os.path.join(ORIG_SAVE_DIR, f"{file_base}_original.png")
            pil_image.save(orig_save_path)

            input_tensor = transform(pil_image).unsqueeze(0).to(active_device)

            # Task
            def run_encode():
                with torch.no_grad():
                    _, _, _, id_t, id_b = model.encode(input_tensor)
                bytes_t = id_t.cpu().numpy().astype('uint16').tobytes()
                bytes_b = id_b.cpu().numpy().astype('uint16').tobytes()
                return id_t.shape, id_b.shape, bytes_t, bytes_b

            # Run & Time
            (shape_t, shape_b, bytes_t, bytes_b), duration = measure_execution(active_device, run_encode)

            # Save Binary
            save_path = os.path.join(COMPRESSED_DIR, f"{file_base}.bin")
            with open(save_path, 'wb') as f:
                f.write(struct.pack('iiii', shape_t[1], shape_t[2], shape_b[1], shape_b[2]))
                f.write(bytes_t)
                f.write(bytes_b)

        elif MODE == 'decompress':
            # Load Binary
            bin_path = os.path.join(COMPRESSED_DIR, filename)
            with open(bin_path, 'rb') as f:
                h_t, w_t, h_b, w_b = struct.unpack('iiii', f.read(16))
                code_t_raw = np.frombuffer(f.read(h_t * w_t * 2), dtype='uint16').reshape(1, h_t, w_t)
                code_b_raw = np.frombuffer(f.read(h_b * w_b * 2), dtype='uint16').reshape(1, h_b, w_b)

            # Task
            def run_decode():
                code_t = torch.from_numpy(code_t_raw.astype(np.int64)).to(active_device)
                code_b = torch.from_numpy(code_b_raw.astype(np.int64)).to(active_device)
                with torch.no_grad():
                    return model.decode_code(code_t, code_b)

            # Run & Time
            reconstructed_tensor, duration = measure_execution(active_device, run_decode)

            # Save Reconstruction
            mean_d = torch.FloatTensor(MEAN_VALS).reshape(1, 3, 1, 1).to(active_device)
            std_d = torch.FloatTensor(STD_VALS).reshape(1, 3, 1, 1).to(active_device)
            recon_tensor = reconstructed_tensor * std_d + mean_d
            recon_tensor = torch.clamp(recon_tensor, 0, 1) # Clip for safety
            
            save_image(recon_tensor, os.path.join(RECON_DIR, f"{file_base}_recon.png"))

        # Log time (Warm-up logic: ignore index 0)
        print(f"[{idx+1}/{len(files)}] Processed {filename} in {duration:.4f}s")
        if idx > 0:
            timings.append(duration)

    # ------------------------------------------------------
    # PHASE 2: CALCULATE METRICS BATCHWISE
    # ------------------------------------------------------
    print(f"\n--- Calculating Metrics for {MODE.upper()} ---")
    
    if MODE == 'compress':
        ratios = []
        total_orig_size = 0
        total_comp_size = 0

        for filename in files:
            file_base = os.path.splitext(filename)[0]
            orig_path = os.path.join(ORIG_SAVE_DIR, f"{file_base}_original.png")
            comp_path = os.path.join(COMPRESSED_DIR, f"{file_base}.bin")
            
            if os.path.exists(orig_path) and os.path.exists(comp_path):
                o_size = os.path.getsize(orig_path)
                c_size = os.path.getsize(comp_path)
                
                # Avoid division by zero
                if c_size > 0:
                    ratios.append(o_size / c_size)
                    total_orig_size += o_size
                    total_comp_size += c_size
        
        # Report
        if ratios:
            avg_ratio = sum(ratios) / len(ratios)
            print(f"Average Compression Ratio per image(original size/compressed size): {avg_ratio:.2f}x")
            print(f"Total Dataset Ratio: {(total_orig_size/total_comp_size):.2f}x")
        else:
            print("No files found to calculate ratio.")

    elif MODE == 'decompress':
        psnr_scores = []
        
        for filename in files:
            file_base = os.path.splitext(filename)[0]
            # Match reconstruction with original
            orig_path = os.path.join(ORIG_SAVE_DIR, f"{file_base}_original.png")
            recon_path = os.path.join(RECON_DIR, f"{file_base}_recon.png") # Note: saved as _recon.png in phase 1

            if os.path.exists(orig_path) and os.path.exists(recon_path):
                # Load as uint8
                img_orig = np.array(Image.open(orig_path).convert('RGB'))
                img_recon = np.array(Image.open(recon_path).convert('RGB'))
                
                val = calculate_psnr(img_orig, img_recon)
                psnr_scores.append(val)
        
        # Report
        if psnr_scores:
            print(f"Average PSNR: {np.mean(psnr_scores):.2f} dB")
        else:
            print("No pairs found to calculate PSNR.")

    # ------------------------------------------------------
    # FINAL TIMING SUMMARY
    # ------------------------------------------------------
    print(f"\n--- Timing Summary ({MODE}) ---")
    if timings:
        avg_time = sum(timings) / len(timings)
        print(f"Average Processing Time (ignoring warm-up): {avg_time:.4f} seconds")
        print(f"Average FPS: {1.0/avg_time:.2f}")
    else:
        print("Not enough images to calculate average timing.")

if __name__ == '__main__':
    run_process()