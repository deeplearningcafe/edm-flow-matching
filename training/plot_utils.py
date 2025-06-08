import matplotlib.pyplot as plt
import torch
from typing import List
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm

def create_image_grid(images: torch.Tensor, gridw: int = 8, gridh: int = 8):
    """
    Create a grid from a batch of images.
    
    Args:
        images: Tensor of shape (B, C, H, W) with values in [-1, 1]
        gridw: Width of the grid (number of images per row)
        gridh: Height of the grid (number of rows)
    
    Returns:
        PIL Image of the grid
    """
    import PIL.Image
    
    batch_size = images.shape[0]
    # images are already float from the generation function
    images = images.float()
    if batch_size < gridw * gridh:
        # Pad with zeros if we don't have enough images
        padding = torch.zeros(
            gridw * gridh - batch_size, 
            *images.shape[1:], 
            device=images.device, 
            dtype=images.dtype
        )
        images = torch.cat([images, padding], dim=0)
    elif batch_size > gridw * gridh:
        # Take only what we need
        images = images[:gridw * gridh]
    
    # Denormalize from [-1, 1] to [0, 255]
    images = (images * 127.5 + 128).clip(0, 255).to(torch.uint8)
    
    # Reshape to grid format
    C, H, W = images.shape[1], images.shape[2], images.shape[3]
    images = images.view(gridh, gridw, C, H, W)
    images = images.permute(0, 3, 1, 4, 2)  # (gridh, H, gridw, W, C)
    images = images.reshape(gridh * H, gridw * W, C)
    
    # Convert to PIL Image
    images = images.float().cpu().numpy()
    if C == 1:  # Grayscale
        images = images.squeeze(-1)
        return PIL.Image.fromarray(images, 'L')
    else:  # RGB
        return PIL.Image.fromarray(images, 'RGB')

def plot_generation_steps_with_grid(
    trajectory: List[torch.Tensor],
    batch_labels: torch.Tensor = None,
    num_classes: int = 10,
    filename: str = "generation_results.png",
    title: str = "Generation Results",
    epoch: int = 0,
    eval_interval: int = 100,
    gridw: int = 8,
    gridh: int = 8
):
    """
    Enhanced plotting function that creates both trajectory and grid plots.
    
    Args:
        trajectory: List of tensors (B, C, H, W) at different steps in [-1, 1] range
        batch_labels: Labels for the batch, used to select one sample per class
        num_classes: Number of classes (for trajectory selection)
        filename: Output filename
        title: Plot title
        epoch: Current epoch number
        eval_interval: Evaluation interval for custom x-axis
        gridw: Grid width for final image grid
        gridh: Grid height for final image grid
    """
    if not trajectory or len(trajectory) == 0:
        print("Trajectory list is empty.")
        return None
    
    # Create figure with subplots: trajectory on top, grid on bottom
    fig = plt.figure(figsize=(16, 12))
    
    # === TRAJECTORY PLOT ===
    # Select one sample per class for trajectory visualization
    trajectory_indices = []
    if batch_labels is not None:
        # Find first occurrence of each class
        for class_id in range(min(num_classes, batch_labels.max().item() + 1)):
            class_mask = (batch_labels == class_id)
            if class_mask.any():
                first_idx = torch.where(class_mask)[0][0].item()
                trajectory_indices.append(first_idx)
    else:
        # If no labels, just take first num_classes samples
        trajectory_indices = list(range(min(num_classes, trajectory[0].shape[0])))
    
    num_traj_samples = len(trajectory_indices)
    num_steps = len(trajectory)
    
    if num_traj_samples > 0 and num_steps > 0:
        # Create subplot for trajectory (top half)
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.3)
        
        # Trajectory subplot
        ax_traj = fig.add_subplot(gs[0])
        
        # Create trajectory grid
        traj_fig, traj_axes = plt.subplots(
            num_traj_samples, num_steps,
            figsize=(num_steps * 1.2, num_traj_samples * 1.2)
        )
        
        # Handle single sample/step cases
        if num_traj_samples == 1 and num_steps == 1:
            traj_axes = np.array([[traj_axes]])
        elif num_traj_samples == 1:
            traj_axes = traj_axes.reshape(1, num_steps)
        elif num_steps == 1:
            traj_axes = traj_axes.reshape(num_traj_samples, 1)
        
        for i, sample_idx in enumerate(trajectory_indices):
            for j in range(num_steps):
                img_tensor = trajectory[j][sample_idx]
                
                # Convert and denormalize
                img = img_tensor.float().cpu().numpy()
                if img.shape[0] == 1:  # Grayscale
                    img = np.squeeze(img, axis=0)
                    img = (img + 1.0) / 2.0
                    img = np.clip(img, 0, 1)
                    cmap = 'gray'
                else:  # RGB
                    img = np.transpose(img, (1, 2, 0))
                    img = (img + 1.0) / 2.0
                    img = np.clip(img, 0, 1)
                    cmap = None
                
                ax = traj_axes[i, j]
                ax.imshow(img, cmap=cmap)
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Add titles and labels
                if i == 0:  # Step labels on top row
                    # Custom x-axis based on eval_interval and epochs
                    if j == 0:
                        step_label = "t=0 (Noise)"
                    elif j == num_steps - 1:
                        step_label = "t=1 (Final)"
                    else:
                        # Calculate approximate step based on position
                        progress = j / (num_steps - 1)
                        step_num = int(progress * 50)  # Assuming 50 inference steps
                        step_label = f"Step ~{step_num}"
                    ax.set_title(step_label, fontsize=9)
                
                if j == 0:  # Class labels on left column
                    class_label = (batch_labels[sample_idx].item() 
                                 if batch_labels is not None 
                                 else f"Sample {sample_idx}")
                    ax.set_ylabel(f"Class {class_label}", fontsize=9)
        
        traj_fig.suptitle(f"Generation Trajectory - Epoch {epoch}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save trajectory plot separately
        traj_filename = filename.replace('.png', '_trajectory.png')
        traj_fig.savefig(traj_filename, dpi=100, bbox_inches='tight')
        plt.close(traj_fig)
        print(f"Saved trajectory plot to {traj_filename}")
    
    # === FINAL IMAGE GRID ===
    if len(trajectory) > 0:
        final_images = trajectory[-1]  # Get final generated images
        
        # Create and save image grid
        grid_image = create_image_grid(final_images, gridw=gridw, gridh=gridh)
        grid_filename = filename.replace('.png', '_grid.png')
        grid_image.save(grid_filename)
        print(f"Saved image grid to {grid_filename}")
        
        # Also create a matplotlib subplot for the grid in the main figure
        ax_grid = fig.add_subplot(gs[1])
        ax_grid.imshow(np.array(grid_image), cmap='gray' if final_images.shape[1] == 1 else None)
        ax_grid.set_title(f"Generated Batch Grid - Epoch {epoch}", fontsize=12)
        ax_grid.set_xticks([])
        ax_grid.set_yticks([])
        
        # Add custom x-axis information
        info_text = f"Epoch: {epoch} | Eval Interval: {eval_interval} | Batch Size: {final_images.shape[0]}"
        ax_grid.text(0.5, -0.1, info_text, transform=ax_grid.transAxes, 
                    ha='center', va='top', fontsize=10)
    
    # Save combined figure
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    try:
        fig.savefig(filename, dpi=100, bbox_inches='tight')
        print(f"Saved combined plot to {filename}")
        plt.close(fig)
        return filename
    except Exception as e:
        print(f"Error saving plot: {e}")
        plt.close(fig)
        return None

def generate_samples_edm_fm(
    unet_model: torch.nn.Module, # Expects a SongUNet instance
    num_samples: int = 10,
    # cond_input for generation (e.g., list of int labels or pre-formatted)
    cond_input: torch.Tensor = None,
    device: str = "cuda",
    num_inference_steps: int = 50,
    img_shape: tuple = (3, 32, 32),
    edm_params: dict = None,
    seed: int = None,
    # unet_label_dim is SongUNet's label_dim
    unet_label_dim: int = 0,
    dtype=torch.float32,
):
    """
    Generates samples using Flow Matching with a SongUNet.
    Uses a simple Euler ODE solver.
    """
    if edm_params is None:
        edm_params = {
            'sigma_min': 0.002, 'sigma_max': 80.0, 'sigma_data': 0.5
        }

    unet_model.eval()
    rand_kwargs = {}
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        rand_kwargs['generator'] = generator

    # Initialize from noise (x0)
    x = torch.randn(num_samples, *img_shape, device=device, dtype=dtype, **rand_kwargs)

    # Time steps for Euler solver (from t=0 to t=1)
    t_steps = torch.linspace(
        0, 1, num_inference_steps + 1, device=device, dtype=dtype
    )
    trajectory = [x.detach().cpu()]

    # Prepare `class_labels` for SongUNet for the entire batch
    unet_class_labels_gen = None
    if unet_label_dim > 0:
        if cond_input is None:
            # If conditional model but no labels, generate random labels
            # or raise error. For now, let's generate random ones.
            print(f"Warning: Conditional model (label_dim={unet_label_dim}) "
                  "but no cond_input provided. Generating random labels.")
            cond_input = torch.randint(
                0, unet_label_dim, (num_samples,),
                device=device, generator=generator
            )
        # Ensure cond_input is a tensor of integer labels for one-hot
        if not isinstance(cond_input, torch.Tensor):
            cond_input_tensor = torch.tensor(
                cond_input, dtype=torch.long, device=device
            )
        else:
            cond_input_tensor = cond_input.to(
                device=device, dtype=torch.long
            )

        # One-hot encode the labels
        unet_class_labels_gen = F.one_hot(
            cond_input_tensor, num_classes=unet_label_dim
        ).to(dtype)
        # Ensure batch size of labels matches num_samples
        if unet_class_labels_gen.shape[0] != num_samples:
            if unet_class_labels_gen.shape[0] == 1 and num_samples > 1:
                print(f"Repeating single cond_input for {num_samples} samples.")
                unet_class_labels_gen = unet_class_labels_gen.repeat(
                    num_samples, 1
                )
            else:
                raise ValueError(
                    f"Batch size of cond_input ({cond_input_tensor.shape[0]}) "
                    f"does not match num_samples ({num_samples}) "
                    f"and cannot be broadcasted."
                )

    with torch.no_grad():
        # for i in tqdm(range(num_inference_steps), desc="Sampling (EDM-FM)"):
        for i in range(num_inference_steps):
            t_current = t_steps[i]
            dt = t_steps[i+1] - t_current

            # Replicate t_current for batch
            t_batch = t_current.expand(num_samples)

            sigma = edm_params['sigma_max'] * \
                    (edm_params['sigma_min'] / edm_params['sigma_max']) ** t_batch

            sigma_b = sigma.view(-1, 1, 1, 1)
            c_in_val = 1 / (edm_params['sigma_data']**2 + sigma_b**2).sqrt()
            unet_noise_labels = (sigma.log() / 4).flatten()

            v = unet_model(
                x=x * c_in_val,
                noise_labels=unet_noise_labels,
                class_labels=unet_class_labels_gen,
                augment_labels=None
            )

            x = x + v * dt # Euler step

            if (i + 1) % max(num_inference_steps // 10, 1) == 0 or \
               i == num_inference_steps - 1:
                trajectory.append(x.detach().float().cpu())

    unet_model.train() # Set back to train mode
    return x.detach().float().cpu(), trajectory

