import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os

# Custom weight initialization function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def create_dataloader(data_dir, image_size, batch_size, loader_workers):
    # Check data directory structure
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory {data_dir} does not exist")
    
    # Check directory contents
    contents = os.listdir(data_dir)
    print(f"Data directory contents: {contents}")
    
    # Check if it's a standard ImageFolder structure
    is_imagefolder = any(os.path.isdir(os.path.join(data_dir, d)) for d in contents)
    
    # Adjust data loader based on directory structure
    if is_imagefolder:
        # Standard ImageFolder structure
        dataset = dset.ImageFolder(root=data_dir,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    else:
        # Check if there are image files
        image_files = [f for f in contents if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            print(f"Found {len(image_files)} image files, creating temporary data structure...")
            # Create temporary class directory
            tmp_dir = os.path.join(data_dir, "_tmp_class")
            if not os.path.exists(tmp_dir):
                os.makedirs(tmp_dir)
            
            # Create symbolic links to images
            for img in image_files:
                src = os.path.join(data_dir, img)
                dst = os.path.join(tmp_dir, img)
                if not os.path.exists(dst):
                    # On Windows, creating symbolic links requires admin privileges, so here we use copy or create hard links instead
                    try:
                        os.link(src, dst)  # Create hard link
                    except:
                        import shutil
                        shutil.copy2(src, dst)  # If hard link fails, copy file instead
            
            dataset = dset.ImageFolder(root=os.path.dirname(tmp_dir),
                                   transform=transforms.Compose([
                                       transforms.Resize(image_size),
                                       transforms.CenterCrop(image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
        else:
            # Check if there are image files
            for root, dirs, files in os.walk(data_dir):
                image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:
                    print(f"Found image files in subdirectory {root}")
                    dataset = dset.ImageFolder(root=os.path.dirname(root),
                                           transform=transforms.Compose([
                                               transforms.Resize(image_size),
                                               transforms.CenterCrop(image_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))
                    break
            else:
                raise FileNotFoundError("Unable to find image files in the data directory")
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=loader_workers)
    
    print(f"Dataset size: {len(dataset)} images, {len(dataloader)} batches")
    return dataloader

# Smooth loss data
def moving_average(data, window_size):
    if len(data) < window_size:
        # If data points are less than window size, return original data
        return np.array(data)
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

def plot_losses(G_losses, D_losses, smooth_amt=5, output_path='loss_plot.png'):
    # Ensure loss lists are not empty
    if not G_losses or not D_losses:
        print("Warning: Loss data is empty, cannot plot")
        # Create blank plot
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(output_path)
        plt.close()
        return
    
    # Adjust smooth window size
    smooth_amt = min(smooth_amt, len(G_losses), len(D_losses))
    if smooth_amt < 2:
        # If window size is too small, do not smooth
        g_smooth = G_losses
        d_smooth = D_losses
    else:
        # Smooth data
        g_smooth = moving_average(G_losses, smooth_amt)
        d_smooth = moving_average(D_losses, smooth_amt)
    
    # Plot
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_smooth, label="Generator")
    plt.plot(d_smooth, label="Discriminator")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    
    # Save image
    plt.savefig(output_path)
    plt.close()

def save_output_img(dataloader, img_list, device, image_size, output_path='output_images.png'):
    if not img_list:
        print("Warning: No generated images, cannot save")
        return
    
    try:
        # Get a batch of real images
        real_batch = next(iter(dataloader))
        
        # Create figure
        plt.figure(figsize=(15,7))
        
        # Plot real images
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.title("Real Images")
        real_grid = vutils.make_grid(real_batch[0].to(device)[:16], padding=2, normalize=True)
        plt.imshow(np.transpose(real_grid.cpu(), (1,2,0)))
        
        # Plot generated images
        plt.subplot(1,2,2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(img_list[-1], (1,2,0)))
        
        # Save image
        plt.savefig(output_path)
        plt.close()
        
        # Save generated image separately
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Generated Images")
        plt.imshow(np.transpose(img_list[-1], (1,2,0)))
        
        # Extract directory from output path
        output_dir = os.path.dirname(output_path)
        if output_dir == '':
            output_dir = '.'
        
        # Save separate generated image
        generated_path = os.path.join(output_dir, 'generated_' + os.path.basename(output_path))
        plt.savefig(generated_path)
        plt.close()
        
    except Exception as e:
        print(f"Error saving image: {e}")
        
        # If error, at least save generated image
        try:
            plt.figure(figsize=(8,8))
            plt.axis("off")
            plt.title("Generated Images Only")
            plt.imshow(np.transpose(img_list[-1], (1,2,0)))
            
            # Save image
            plt.savefig(output_path.replace('.png', '_gen_only.png'))
            plt.close()
        except:
            print("Unable to save generated image")