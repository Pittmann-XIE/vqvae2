import argparse
import os
import glob
import h5py
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Convert raw images to HDF5 for VQ-VAE")
    parser.add_argument('--dataset_name', type=str, default='COCO', 
                        help='Name of dataset to use in filename (e.g., COCO, CheXpert)')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Root directory containing train2017 and val2017 folders')
    parser.add_argument('--output_dir', type=str, default='./HDF5_datasets',
                        help='Where to save the .hdf5 files')
    parser.add_argument('--img_size', type=int, default=256, help='Size to resize image to')
    parser.add_argument('--crop_size', type=int, default=256, help='Size to center crop')
    parser.add_argument('--view', type=str, default='frontal', 
                        help='Dummy view name to satisfy filename requirements (frontal/lateral)')
    return parser.parse_args()

def process_data(args, mode, folder_name):
    """
    Args:
        mode: 'train' or 'valid' (naming convention for output)
        folder_name: 'train2017' or 'val2017' (actual folder name)
    """
    
    # Define paths
    src_path = os.path.join(args.data_dir, folder_name)
    if not os.path.exists(src_path):
        print(f"Warning: Folder {src_path} does not exist. Skipping...")
        return

    # 1. Collect all images (support .png as requested, added .jpg just in case)
    image_paths = glob.glob(os.path.join(src_path, '*.png')) + \
                  glob.glob(os.path.join(src_path, '*.jpg')) + \
                  glob.glob(os.path.join(src_path, '*.jpeg'))
    
    image_paths.sort() # Ensure deterministic order
    num_images = len(image_paths)
    
    if num_images == 0:
        print(f"No images found in {src_path}")
        return

    print(f"Found {num_images} images in {folder_name}. Converting to HDF5...")

    # 2. Define Transforms (Matching example.py)
    nc = 3  # RGB
    mean = [0.485, 0.456, 0.406] 
    std = [0.229, 0.224, 0.225]
    
    transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # 3. Create HDF5 File
    # Filename format: {dataset}_{mode}_{size}_{view}_normalized.hdf5
    filename = f"{args.dataset_name}_{mode}_{args.crop_size}_{args.view}_normalized.hdf5"
    output_path = os.path.join(args.output_dir, filename)
    
    # Create directory if needed
    os.makedirs(args.output_dir, exist_ok=True)

    # 4. Write Data
    # We use a context manager to ensure file is closed properly
    with h5py.File(output_path, 'w') as hdf5:
        
        # Create datasets inside HDF5
        # Shape: (N, Channels, H, W)
        img_ds = hdf5.create_dataset('img', (num_images, nc, args.crop_size, args.crop_size), dtype='f')
        
        # Create dummy labels. train.py expects a 'labels' key, but VQ-VAE ignores the content.
        # We create a placeholder of shape (N, 14) to match the dimensions seen in your example.py
        # If your model doesn't care about dimension 14, (N, 1) works too, but 14 is safer based on provided code.
        num_dummy_labels = 14 
        lbl_ds = hdf5.create_dataset('labels', (num_images, num_dummy_labels), dtype='f')

        # Loop and process
        for i, path in enumerate(tqdm(image_paths, desc=f"Processing {mode}")):
            try:
                # Load image
                with open(path, 'rb') as f:
                    img = Image.open(f)
                    img = img.convert('RGB') # Ensure 3 channels
                
                # Apply transform
                img_tensor = transform(img)
                
                # Write to HDF5
                img_ds[i, ...] = img_tensor.numpy()
                lbl_ds[i, ...] = np.zeros(num_dummy_labels) # Dummy label
                
            except Exception as e:
                print(f"Error processing {path}: {e}")

    print(f"Saved: {output_path}")

def main():
    args = get_args()
    
    # Map the actual folder names to the modes expected by train.py
    # train.py uses 'train' and 'valid' strings for loading
    folders_to_process = [
        ('train', 'train2017'),
        ('valid', 'val2017')
    ]

    for mode, folder in folders_to_process:
        process_data(args, mode, folder)

if __name__ == '__main__':
    main()