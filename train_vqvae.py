# import argparse
# import os
# import numpy as np
# import torch
# from torch import nn, optim
# from torch.utils.data import DataLoader
# from torch.autograd import Variable
# from tqdm import tqdm
# from networks import VQVAE
# from utilities import ChestXrayHDF5, recon_image, save_loss_plots

# cuda = True if torch.cuda.is_available() else False
# Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# parser = argparse.ArgumentParser()
# parser.add_argument('--size', type=int, default=256)
# parser.add_argument('--n_epochs', type=int, default=5600)
# parser.add_argument('--lr', type=float, default=3e-4)
# parser.add_argument('--first_stride', type=int, default=4, help="2, 4, 8, or 16")
# parser.add_argument('--second_stride', type=int, default=2, help="2, 4, 8, or 16")
# parser.add_argument('--embed_dim', type=int, default=64)
# parser.add_argument('--data_path', type=str, default='./HDF5_datasets')
# parser.add_argument('--dataset', type=str, default='COCO', help="CheXpert or mimic or COCO")
# parser.add_argument('--view', type=str, default='frontal', help="frontal or lateral")
# parser.add_argument('--save_path', type=str, default='./checkpoints')
# parser.add_argument('--train_run', type=str, default='0')
# args = parser.parse_args()
# torch.manual_seed(816)

# save_path = f'{args.save_path}/{args.dataset}/{args.train_run}'
# os.makedirs(save_path, exist_ok=True)
# os.makedirs(f'{save_path}/checkpoint/', exist_ok=True)
# os.makedirs(f'{save_path}/sample/', exist_ok=True)
# with open(f'{save_path}/args.txt', 'w') as f:
#     for key in vars(args).keys():
#         f.write(f'{key}: {vars(args)[key]}\n')
#         print(f'{key}: {vars(args)[key]}')

# dataloaders = {}
# dataloaders['train'] = DataLoader(ChestXrayHDF5(f'{args.data_path}/{args.dataset}_train_{args.size}_{args.view}_normalized.hdf5'),
#                                   batch_size=128,
#                                   shuffle=True,
#                                   drop_last=True)
# dataloaders['valid'] = DataLoader(ChestXrayHDF5(f'{args.data_path}/{args.dataset}_valid_{args.size}_{args.view}_normalized.hdf5'),
#                                   batch_size=128,
#                                   shuffle=True,
#                                   drop_last=True)
# for i, (img, targets) in enumerate(dataloaders['valid']):
#     sample_img = Variable(img.type(Tensor))
#     break


# if cuda:
#     model = VQVAE(first_stride=args.first_stride, second_stride=args.second_stride, embed_dim=args.embed_dim).cuda()
# else:
#     model = VQVAE(first_stride=args.first_stride, second_stride=args.second_stride, embed_dim=args.embed_dim)
# n_gpu = torch.cuda.device_count()
# if n_gpu > 1:
#     device_ids = list(range(n_gpu))
#     model = nn.DataParallel(model, device_ids=device_ids)

# optimizer = optim.Adam(model.parameters(), lr=args.lr)

# losses = np.zeros((2, args.n_epochs, 3))  # [0,:,:] index for train, [1,:,:] index for valid

# for epoch in range(args.n_epochs):
#     for phase in ['train', 'valid']:
#         model.train(phase == 'train')
#         criterion = nn.MSELoss()

#         latent_loss_weight = 0.25
#         n_row = 5
#         loader = tqdm(dataloaders[phase])
        
#         # --- LOOP STARTS ---
#         for i, (img, label) in enumerate(loader):
#             img = Variable(img.type(Tensor))
#             with torch.set_grad_enabled(phase == 'train'):
#                 optimizer.zero_grad()
#                 out, latent_loss = model(img)
#                 recon_loss = criterion(out, img)
#                 latent_loss = latent_loss.mean()
#                 loss = recon_loss + latent_loss_weight * latent_loss
                
#                 if phase == 'train':
#                     loss.backward()
#                     optimizer.step()
#                     losses[0, epoch, :] = [loss.item(), recon_loss.item(), latent_loss.item()] 
#                 else:
#                     losses[1, epoch, :] = [loss.item(), recon_loss.item(), latent_loss.item()]
#                 lr = optimizer.param_groups[0]['lr']

#             # REMOVED: loader.set_description(...)

#             if i % 20 == 0:
#                 recon_image(n_row, sample_img, model, f'{save_path}', epoch, Tensor)
#         # --- LOOP ENDS ---

#         # NEW: Print only once per phase per epoch
#         print(f'phase: {phase}; epoch: {epoch + 1}; total_loss: {loss.item():.5f}; '
#               f'latent: {latent_loss.item():.5f}; mse: {recon_loss.item():.5f}; '
#               f'lr: {lr:.5f}')

#         save_loss_plots(args.n_epochs, epoch, losses, f'{save_path}')
#     if (epoch + 1) % 20 == 0:
#         torch.save(model.state_dict(), f'{save_path}/checkpoint/vqvae_{str(epoch + 1).zfill(3)}.pt')


## continue training + dropped learning rate
import argparse
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from collections import OrderedDict
from networks import VQVAE
from utilities import ChestXrayHDF5, recon_image, save_loss_plots

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

parser = argparse.ArgumentParser()
parser.add_argument('--size', type=int, default=256)
parser.add_argument('--n_epochs', type=int, default=5600)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--first_stride', type=int, default=4, help="2, 4, 8, or 16")
parser.add_argument('--second_stride', type=int, default=2, help="2, 4, 8, or 16")
parser.add_argument('--embed_dim', type=int, default=64)
parser.add_argument('--data_path', type=str, default='./HDF5_datasets')
parser.add_argument('--dataset', type=str, default='COCO', help="CheXpert or mimic or COCO")
parser.add_argument('--view', type=str, default='frontal', help="frontal or lateral")
parser.add_argument('--save_path', type=str, default='./checkpoints')
parser.add_argument('--train_run', type=str, default='0')
parser.add_argument('--resume', type=str, default='/home/xie/vqvae2/checkpoints/COCO/0/checkpoint/vqvae_020.pt', help="path to checkpoint to resume from")
args = parser.parse_args()
torch.manual_seed(816)

save_path = f'{args.save_path}/{args.dataset}/{args.train_run}'
os.makedirs(save_path, exist_ok=True)
os.makedirs(f'{save_path}/checkpoint/', exist_ok=True)
os.makedirs(f'{save_path}/sample/', exist_ok=True)

dataloaders = {}
dataloaders['train'] = DataLoader(ChestXrayHDF5(f'{args.data_path}/{args.dataset}_train_{args.size}_{args.view}_normalized.hdf5'),
                                  batch_size=128, shuffle=True, drop_last=True)
dataloaders['valid'] = DataLoader(ChestXrayHDF5(f'{args.data_path}/{args.dataset}_valid_{args.size}_{args.view}_normalized.hdf5'),
                                  batch_size=128, shuffle=True, drop_last=True)

for i, (img, targets) in enumerate(dataloaders['valid']):
    sample_img = Variable(img.type(Tensor))
    break

# Initialize Model and Optimizer
model = VQVAE(first_stride=args.first_stride, second_stride=args.second_stride, embed_dim=args.embed_dim)
if cuda:
    model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr)

# NEW: Initialize Scheduler
# This will reduce LR by half if valid loss doesn't improve for 10 epochs
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)

if args.resume:
    if os.path.isfile(args.resume):
        print(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location='cpu' if not cuda else None)
        
        # Load Model weights (handling DataParallel prefix)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k 
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        
        # Load Optimizer and Scheduler states if they exist
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Determine start epoch
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        else:
            try:
                start_epoch = int(os.path.basename(args.resume).split('_')[-1].split('.')[0])
            except ValueError:
                start_epoch = 0
    else:
        print(f"=> no checkpoint found at '{args.resume}'")

n_gpu = torch.cuda.device_count()
if n_gpu > 1:
    model = nn.DataParallel(model)

losses = np.zeros((2, args.n_epochs, 3)) 

for epoch in range(start_epoch, args.n_epochs):
    epoch_val_loss = 0 # To track for scheduler
    
    for phase in ['train', 'valid']:
        model.train(phase == 'train')
        criterion = nn.MSELoss()
        latent_loss_weight = 0.25
        n_row = 5
        loader = tqdm(dataloaders[phase], desc=f"Epoch {epoch+1}/{args.n_epochs} [{phase}]")
        
        running_loss = 0.0
        for i, (img, label) in enumerate(loader):
            img = Variable(img.type(Tensor))
            with torch.set_grad_enabled(phase == 'train'):
                optimizer.zero_grad()
                out, latent_loss = model(img)
                recon_loss = criterion(out, img)
                latent_loss = latent_loss.mean()
                loss = recon_loss + latent_loss_weight * latent_loss
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    losses[0, epoch, :] = [loss.item(), recon_loss.item(), latent_loss.item()] 
                else:
                    losses[1, epoch, :] = [loss.item(), recon_loss.item(), latent_loss.item()]
                
                running_loss += loss.item()

            if i % 20 == 0:
                recon_image(n_row, sample_img, model, f'{save_path}', epoch, Tensor)

        phase_loss = running_loss / len(loader)
        if phase == 'valid':
            epoch_val_loss = phase_loss
            # NEW: Step the scheduler based on validation loss
            scheduler.step(epoch_val_loss)

        lr = optimizer.param_groups[0]['lr']
        print(f'phase: {phase}; epoch: {epoch + 1}; avg_loss: {phase_loss:.5f}; lr: {lr:.6f}')
        save_loss_plots(args.n_epochs, epoch, losses, f'{save_path}')
        
    if (epoch + 1) % 20 == 0:
        # Save a comprehensive dictionary
        state_to_save = {
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict() if n_gpu > 1 else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'losses': losses,
        }
        torch.save(state_to_save, f'{save_path}/checkpoint/vqvae_{str(epoch + 1).zfill(3)}.pt')