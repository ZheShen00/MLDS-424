import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import yaml
from generator import Generator
from discriminator import Discriminator

# Custom weight initialization function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Create a custom dataset class to better handle Google's Cartoonset
class CartoonDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the cartoon images.
            transform (callable, optional): Transform to be applied on images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        
        # Look for PNG files in the root directory
        for img_file in glob.glob(os.path.join(root_dir, "*.png")):
            self.image_files.append(img_file)
            
        # If no PNG files found directly, look in subdirectories
        if len(self.image_files) == 0:
            for subdir in os.listdir(root_dir):
                subdir_path = os.path.join(root_dir, subdir)
                if os.path.isdir(subdir_path):
                    for img_file in glob.glob(os.path.join(subdir_path, "*.png")):
                        self.image_files.append(img_file)
        
        print(f"Found {len(self.image_files)} cartoon images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            
        return image, 0  # Return 0 as a dummy label

def main():
    # Set random seed for reproducibility
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    print("Random Seed:", manualSeed)

    # Load configuration
    try:
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)
    except:
        print("Error loading config.yaml, using default values")
        config = {
            'data_dir': "cartoonset10k",
            'workers': 4,
            'batch_size': 128,
            'image_size': 64,
            'nc': 3,
            'nz': 100,
            'ngf': 64,
            'ndf': 64,
            'num_epochs': 20,  # Reduced to 20 epochs
            'lr': 0.0002,
            'beta1': 0.5,
            'ngpu': 1,
            'save_interval': 250,  # Reduced save interval
            'label_smoothing': 0.9,
            'weight_init_mean': 0.0,
            'weight_init_std': 0.02,
            'use_spectral_norm': True,  # New parameters with defaults
            'use_dropout': True,
            'use_diff_lr': True,
            'diff_lr_factor': 0.8
        }
        
    # Extract parameters from config
    dataroot = config['data_dir']
    workers = config['workers']
    batch_size = config['batch_size']
    image_size = config['image_size']
    nc = config['nc']
    nz = config['nz']
    ngf = config['ngf']
    ndf = config['ndf']
    num_epochs = config['num_epochs']
    lr = config['lr']
    beta1 = config['beta1']
    ngpu = config['ngpu']
    save_interval = config.get('save_interval', 250)
    label_smoothing = config.get('label_smoothing', 0.9)
    weight_init_mean = config.get('weight_init_mean', 0.0)
    weight_init_std = config.get('weight_init_std', 0.02)
    
    # Extract new advanced parameters
    use_spectral_norm = config.get('use_spectral_norm', True)
    use_dropout = config.get('use_dropout', True)
    use_diff_lr = config.get('use_diff_lr', True)
    diff_lr_factor = config.get('diff_lr_factor', 0.8)

    # Create output directory
    os.makedirs("results", exist_ok=True)

    # Set device
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print("Device:", device)

    # Create the transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Create the dataset and dataloader
    try:
        # Try custom dataset first
        dataset = CartoonDataset(dataroot, transform=transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=workers
        )
        print(f"Dataset loaded with {len(dataset)} images")
        
        # Sanity check - verify image dimensions
        sample_batch = next(iter(dataloader))
        print(f"Sample batch shape: {sample_batch[0].shape}")
        
        # Save some real examples
        real_samples = vutils.make_grid(sample_batch[0][:64], padding=2, normalize=True, nrow=8)
        plt.figure(figsize=(10,10))
        plt.axis("off")
        plt.title("Real Cartoon Samples")
        plt.imshow(np.transpose(real_samples.cpu().numpy(), (1,2,0)))
        plt.savefig('results/real_samples.png')
        plt.close()
        
    except Exception as e:
        print(f"Error with custom dataset, trying ImageFolder: {e}")
        try:
            # Fallback to ImageFolder
            dataset = dset.ImageFolder(root=dataroot, transform=transform)
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, num_workers=workers
            )
            print(f"ImageFolder dataset loaded with {len(dataset)} images")
        except Exception as e:
            print(f"Failed to load dataset: {e}")
            return

    # Create the generator
    dim_args_G = {
        "random_dim": nz, 
        "gen_dim": ngf, 
        "num_channels": nc,
        "use_spectral_norm": use_spectral_norm
    }
    netG = Generator(ngpu, dim_args=dim_args_G).to(device)

    # Multi-GPU setup if available
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply weight initialization
    def weights_init_custom(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, weight_init_mean, weight_init_std)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, weight_init_std)
            nn.init.constant_(m.bias.data, 0)
    
    netG.apply(weights_init_custom)
    print("Generator architecture:")
    print(netG)

    # Create the discriminator
    dim_args_D = {
        "disc_dim": ndf, 
        "num_channels": nc,
        "use_spectral_norm": use_spectral_norm,
        "use_dropout": use_dropout
    }
    netD = Discriminator(ngpu, dim_args=dim_args_D).to(device)

    # Multi-GPU setup if available
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply weight initialization
    netD.apply(weights_init_custom)
    print("Discriminator architecture:")
    print(netD)

    # Initialize BCELoss function
    criterion = nn.BCELoss()
    
    # Add L1 loss for diversity regularization
    l1_criterion = nn.L1Loss()

    # Create fixed noise for generating samples throughout training
    fixed_noise = torch.randn(64, nz, device=device)

    # Set real and fake labels with smoothing
    real_label = label_smoothing
    fake_label = 0.0

    # Setup optimizers with differential learning rates
    initial_lr_g = lr
    initial_lr_d = lr * diff_lr_factor if use_diff_lr else lr
    
    optimizerD = optim.Adam(netD.parameters(), lr=initial_lr_d, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=initial_lr_g, betas=(beta1, 0.999))
    
    print(f"Initial learning rates - G: {initial_lr_g}, D: {initial_lr_d}")

    # Training loop
    img_list = []
    G_losses = []
    D_losses = []
    D_x_list = []
    D_G_z1_list = []
    D_G_z2_list = []
    iters = 0
    best_loss = float('inf')

    print("Starting Training Loop...")
    
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            
            # Train with real batch
            netD.zero_grad()
            # Format real batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu)
            # Calculate loss on real batch
            errD_real = criterion(output, label)
            # Calculate gradients
            errD_real.backward()
            D_x = output.mean().item()

            # Train with fake batch
            # Generate fake image batch with G
            noise = torch.randn(b_size, nz, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify the fake batch with D
            output = netD(fake.detach())
            # Calculate D's loss on the fake batch
            errD_fake = criterion(output, label)
            # Calculate gradients
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            
            # Further refined balancing strategy with special handling for later epochs
            update_ratio = errG.item() if 'errG' in locals() else 1.0
            update_ratio = update_ratio / (errD.item() + 1e-8)
            
            # Adjusted thresholds to better balance G and D in later epochs
            dx_threshold = 0.9 - 0.2 * min(1.0, epoch / 3)  # Less aggressive decrease (0.3→0.2)
            dgz_threshold = 0.1 + 0.2 * min(1.0, epoch / 3)  # Less aggressive increase (0.3→0.2)
            ratio_threshold = 0.3 + 0.1 * min(1.0, epoch / 3)  # Lower starting threshold (0.4→0.3)
            
            # Discriminator conditions - slightly relaxed to help generator
            discriminator_too_strong = D_x > dx_threshold and D_G_z1 < dgz_threshold
            discriminator_too_weak = D_G_z1 > 0.65 and errD.item() > 1.4  # Slightly lower thresholds (0.7→0.65, 1.5→1.4)
            
            # Increase update frequency as training progresses - unchanged
            update_frequency = max(1, 4 - min(3, epoch // 3))
            
            # Warm-up phase for the first few epochs
            if epoch < 3:
                # Regular warm-up logic - unchanged
                force_update = (i % 5 == 0)
                if force_update or not (D_x > 0.9 and D_G_z1 < 0.1):
                    optimizerD.step()
                elif i % 50 == 0:
                    print(f"Warm-up phase: Skipping D update: D_x={D_x:.4f}, D_G_z1={D_G_z1:.4f}")
            elif epoch >= 8:  # Special handling for later epochs (8+)
                # Modified update rule for later epochs to prevent D from becoming too strong
                if i % 3 == 0:  # Reduced frequency (every 3rd batch instead of every 2nd)
                    optimizerD.step()
                elif discriminator_too_weak:
                    optimizerD.step()
                    if i % 50 == 0:
                        print(f"Recovery update for weak D: D_x={D_x:.4f}, D_G_z1={D_G_z1:.4f}")
                elif discriminator_too_strong and update_ratio < 0.4:  # Less strict condition (0.3→0.4)
                    if i % 50 == 0:
                        print(f"Late-stage skip: D_x={D_x:.4f}, D_G_z1={D_G_z1:.4f}, ratio={update_ratio:.4f}")
                else:
                    optimizerD.step()
            else:
                # Middle epochs (3-7)
                if discriminator_too_weak:
                    optimizerD.step()
                    if i % 50 == 0:
                        print(f"Recovery update for weak D: D_x={D_x:.4f}, D_G_z1={D_G_z1:.4f}")
                elif discriminator_too_strong or update_ratio < ratio_threshold or i % update_frequency != 0:
                    if i % 50 == 0:
                        print(f"Skipping D update: D_x={D_x:.4f}, D_G_z1={D_G_z1:.4f}, ratio={update_ratio:.4f}")
                else:
                    optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            
            # Dynamic label smoothing for generator
            cur_label_smooth = real_label
            if epoch >= 10:
                # Gradually reduce smoothing as training progresses
                cur_label_smooth = min(real_label + 0.05 * (epoch - 10) / 10, 0.95)
                
            label.fill_(cur_label_smooth)  # fake labels are real for generator cost
            
            # Forward pass of fake batch through D
            output = netD(fake)
            
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            
            # Add diversity loss in later epochs to prevent mode collapse
            if epoch >= 5:
                # Calculate batch diversity by measuring variance across the batch
                fake_features = fake.view(b_size, -1)
                batch_variance = torch.var(fake_features, dim=0).mean()
                
                # Add diversity loss term (higher variance is better)
                diversity_weight = 0.05  # Control the strength of diversity loss
                diversity_loss = 1.0 / (batch_variance + 1e-8) * diversity_weight
                errG = errG + diversity_loss
                
                if i % 50 == 0:
                    print(f'Diversity loss: {diversity_loss.item():.4f}')
            
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            
            # Adaptive generator updates based on loss
            num_g_updates = 1  # Default is single update
            
            # If generator is struggling (high loss), give it extra updates in later epochs
            if epoch >= 8 and errG.item() > 3.0:
                num_g_updates = 2
            elif epoch >= 15 and errG.item() > 2.0:
                num_g_updates = 3
            
            # Update generator
            optimizerG.step()
            
            # Additional generator updates if needed
            for g_update in range(num_g_updates - 1):
                # Use fresh noise for each update to prevent overfitting
                fresh_noise = torch.randn(b_size, nz, device=device)
                fresh_fake = netG(fresh_noise)
                
                # Zero gradients and compute loss
                netG.zero_grad()
                output = netD(fresh_fake)
                errG_extra = criterion(output, label)
                
                # Add diversity loss for extra updates too
                if epoch >= 5:
                    fresh_fake_features = fresh_fake.view(b_size, -1)
                    fresh_batch_variance = torch.var(fresh_fake_features, dim=0).mean()
                    fresh_diversity_loss = 1.0 / (fresh_batch_variance + 1e-8) * diversity_weight
                    errG_extra = errG_extra + fresh_diversity_loss
                
                errG_extra.backward()
                optimizerG.step()
                
                if i % 50 == 0:
                    print(f'Extra G update {g_update+1}: Loss_G_extra: {errG_extra.item():.4f}')
            
            # Save losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            D_x_list.append(D_x)
            D_G_z1_list.append(D_G_z1)
            D_G_z2_list.append(D_G_z2)
            
            # Output training stats
            if i % 50 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                      f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}')
                
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % save_interval == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True, nrow=8))
                
                # Save current generator samples
                plt.figure(figsize=(10,10))
                plt.axis("off")
                plt.title(f"Fake Images at Epoch {epoch} Iteration {iters}")
                plt.imshow(np.transpose(img_list[-1],(1,2,0)))
                plt.savefig(f'results/fake_images_epoch_{epoch}_iter_{iters}.png')
                plt.close()
                
                # Plot side by side comparison
                plt.figure(figsize=(15,10))
                plt.subplot(1,2,1)
                plt.axis("off")
                plt.title("Real Images")
                if len(dataloader) > 0:
                    real_batch = next(iter(dataloader))
                    plt.imshow(np.transpose(vutils.make_grid(
                        real_batch[0][:64], padding=5, normalize=True, nrow=8).cpu(),(1,2,0)))
                
                plt.subplot(1,2,2)
                plt.axis("off")
                plt.title("Fake Images")
                plt.imshow(np.transpose(img_list[-1],(1,2,0)))
                plt.savefig(f'results/comparison_epoch_{epoch}_iter_{iters}.png')
                plt.close()
                
                # Save model checkpoint
                current_loss = errG.item()
                if current_loss < best_loss:
                    best_loss = current_loss
                    torch.save({
                        'epoch': epoch,
                        'netG_state_dict': netG.state_dict(),
                        'netD_state_dict': netD.state_dict(),
                        'optimizerG_state_dict': optimizerG.state_dict(),
                        'optimizerD_state_dict': optimizerD.state_dict(),
                        'loss_G': errG.item(),
                        'loss_D': errD.item(),
                    }, f'results/best_model.pt')
                
                # Regular checkpoint saving
                torch.save({
                    'epoch': epoch,
                    'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                }, f'results/model_checkpoint_latest.pt')
                
            iters += 1

        # Learning rate adjustment with phase-based approach
        if epoch == 4:  # First adjustment at epoch 5
            # Reduce D's learning rate more to help G catch up
            for param_group in optimizerD.param_groups:
                param_group['lr'] = param_group['lr'] * 0.7  # More aggressive reduction for D
            for param_group in optimizerG.param_groups:
                param_group['lr'] = param_group['lr'] * 0.9  # Slight reduction for G
            print(f"Epoch {epoch+1}: Adjusted learning rates - G: {optimizerG.param_groups[0]['lr']}, D: {optimizerD.param_groups[0]['lr']}")
            
        elif epoch == 9:  # Second adjustment at epoch 10
            # Further reduce D's learning rate to prevent it from overpowering G
            for param_group in optimizerD.param_groups:
                param_group['lr'] = param_group['lr'] * 0.6
            for param_group in optimizerG.param_groups:
                param_group['lr'] = param_group['lr'] * 0.8
            print(f"Epoch {epoch+1}: Adjusted learning rates - G: {optimizerG.param_groups[0]['lr']}, D: {optimizerD.param_groups[0]['lr']}")
            
        elif epoch == 14:  # Third adjustment at epoch 15
            # Final adjustment with stronger reduction for D
            for param_group in optimizerD.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5  # Significant reduction for D
            for param_group in optimizerG.param_groups:
                param_group['lr'] = param_group['lr'] * 0.7  # Less reduction for G
            print(f"Epoch {epoch+1}: Final learning rate adjustment - G: {optimizerG.param_groups[0]['lr']}, D: {optimizerD.param_groups[0]['lr']}")

    # Plot and save loss curves
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="G")
    plt.plot(D_losses, label="D")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('results/loss_plot.png')
    plt.close()
    
    # Plot and save discriminator scores
    plt.figure(figsize=(10,5))
    plt.title("D(x) and D(G(z)) During Training")
    plt.plot(D_x_list, label="D(x)")
    plt.plot(D_G_z1_list, label="D(G(z1))")
    plt.plot(D_G_z2_list, label="D(G(z2))")
    plt.xlabel("Iterations")
    plt.ylabel("Probability")
    plt.legend()
    plt.savefig('results/discriminator_scores_plot.png')
    plt.close()

    print("Training complete!")

if __name__ == '__main__':
    # Support for Windows multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    main()