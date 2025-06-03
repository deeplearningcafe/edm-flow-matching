import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
import os
from datetime import datetime
import numpy as np # For SongUNet's np.sqrt
import re # For layer name matching in LR grouping
from typing import List, Dict
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
try:
    # For ImageFolderDataset used in FID calculation
    from training import dataset as edm_dataset
except ImportError:
    print("Warning: training.dataset (from EDM) not found. "
          "FID calculation will be disabled if it relies on this.")
    edm_dataset = None # Set to None if not found
try:
    from . import logging_utils # if logging_utils is in the same package
except ImportError:
    import logging_utils # if logging_utils is a top-level module
import wandb
import tempfile
import shutil
from PIL import Image
import scipy.linalg

def plot_generation_steps(
    trajectory: List[torch.Tensor], # List of tensors (B, C, H, W) at different steps in [-1, 1] range
    num_images_to_show: int = 8, # How many samples from the batch to display
    filename: str = "generation_trajectory.png",
    title: str = "Generation Trajectory"
):
    """Plots the generation process by showing images at different steps."""
    if not trajectory:
        print("Trajectory list is empty.")
        return None

    num_steps = len(trajectory)
    if num_steps == 0:
        print("Trajectory contains no steps.")
        return
    # Ensure we don't try to show more images than available in batch
    num_images_to_show = min(num_images_to_show, trajectory[0].shape[0])
    if num_images_to_show == 0:
        print("No images to show (batch size might be 0 or num_images_to_show=0).")
        return

    # Create a grid: rows=num_images_to_show, cols=num_steps
    fig, axes = plt.subplots(
        num_images_to_show,
        num_steps,
        figsize=(num_steps * 1.5, num_images_to_show * 1.5)
    )

    # Handle single image/step case for axes indexing, ensure axes is always 2D array
    if num_images_to_show == 1 and num_steps == 1:
        axes = np.array([[axes]])
    elif num_images_to_show == 1:
        axes = axes.reshape(1, num_steps)
    elif num_steps == 1:
        axes = axes.reshape(num_images_to_show, 1)


    for i in range(num_images_to_show): # Iterate through samples in batch
        for j in range(num_steps): # Iterate through time steps
            img_tensor = trajectory[j][i] # Get i-th image at j-th step
            # --- CHANGE: Denormalization happens HERE, right before plotting ---
            # Convert tensor to numpy: (C, H, W) -> (H, W, C) or (H, W)
            # And denormalize from [-1, 1] to [0, 1]
            img = img_tensor.numpy()
            if img.shape[0] == 1: # Grayscale: (1, H, W) -> (H, W)
                img = np.squeeze(img, axis=0)
                # Denormalize: (img + 1) / 2
                img = (img + 1.0) / 2.0
                img = np.clip(img, 0, 1) # Clip to ensure valid range
                cmap = 'gray'
            else: # RGB: (C, H, W) -> (H, W, C)
                img = np.transpose(img, (1, 2, 0))
                # Denormalize: (img + 1) / 2
                img = (img + 1.0) / 2.0
                img = np.clip(img, 0, 1) # Clip to ensure valid range
                cmap = None # Use default RGB colormap handling

            ax = axes[i, j]
            ax.imshow(img, cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0: # Add step title only for the first row
                # Determine step number (approximate based on saved steps)
                total_inference_steps = 50 # Assuming this was the input to generate_samples_fm
                save_interval = max(1, total_inference_steps // 10)
                step_num_approx = (j-1) * save_interval if j > 0 else 0
                if j == 0: step_label = "t=0 (Noise)"
                elif j == num_steps - 1: step_label = f"t=1 (Final)" # Step {total_inference_steps}
                else: step_label = f"Step ~{step_num_approx+1}"

                ax.set_title(step_label, fontsize=8)


    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
    try:
        plt.savefig(filename)
        print(f"Saved generation trajectory plot to {filename}")
        plt.close(fig) # Close the figure to free memory
        return filename # Return the path to the saved image
    except Exception as e:
        print(f"Error saving plot: {e}")
        plt.close(fig)
        return None

def sample_lognorm_timesteps(
    batch_size: int,
    mean: float = 0.0,
    std: float = 1.0,
    device: torch.device = None,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Samples timesteps from [0, 1] using Log-Normal distribution."""
    # Sample from N(mean, std^2) in the logit space
    logit_t = torch.normal(
        mean=mean, std=std, size=(batch_size,),
        device=device, generator=generator
    )
    # Apply sigmoid to map to [0, 1]
    t = torch.sigmoid(logit_t)
    return t


def flow_matching_step(
    unet_model: torch.nn.Module, # Expects a SongUNet instance
    x1: torch.Tensor,
    # cond_input is raw conditioning from data loader (e.g., int labels)
    cond_input: torch.Tensor,
    generator: torch.Generator = None,
    input_perturbation: bool = False,
    edm_params: dict = None,
    # unet_label_dim is SongUNet's label_dim, needed for one-hot encoding
    unet_label_dim: int = 0,
    dtype=torch.float32,
):
    """
    Performs a single training step for Flow Matching using a SongUNet.
    The SongUNet is trained to predict the velocity directly.

    Args:
        unet_model: The raw SongUNet model.
        x1: Batch of target images, shape (B, C, H, W).
        cond_input: Raw conditioning data from the loader (e.g., integer
                    class labels (B,) or pre-computed embeddings).
        generator: Optional random number generator.
        input_perturbation: Whether to add perturbation to interpolated points.
        edm_params: Dict with EDM parameters (sigma_min, sigma_max, sigma_data).
        unet_label_dim: The `label_dim` of the SongUNet, for formatting
                        `class_labels`.
    """
    if edm_params is None:
        edm_params = {
            'sigma_min': 0.002, 'sigma_max': 80.0, 'sigma_data': 0.5
        }

    batch_size = x1.shape[0]
    device = x1.device

    x0 = torch.randn_like(x1, dtype=x1.dtype) # Source noise

    # Sample FM time t uniformly from [0, 1]
    # no need to convert the dtype as the mult has no problems
    t = sample_lognorm_timesteps(batch_size, 0, 1, device, generator)

    # Interpolate: xt = (1-t)x0 + tx1
    xt = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1

    if input_perturbation:
        noise_scale = (1 - t.view(-1, 1, 1, 1)) * 0.1
        xt = xt + noise_scale * torch.randn_like(xt)

    v_true = x1 - x0 # True velocity (U-Net target)

    # Convert FM time `t` to EDM's `sigma`
    sigma = edm_params['sigma_max'] * \
            (edm_params['sigma_min'] / edm_params['sigma_max']) ** t

    # Calculate EDM U-Net input conditioning
    sigma_b = sigma.view(-1, 1, 1, 1) # (B, 1, 1, 1)
    c_in_val = 1 / (edm_params['sigma_data']**2 + sigma_b**2).sqrt()
    # `noise_labels` for SongUNet
    unet_noise_labels = (sigma.log() / 4).flatten()

    # Prepare `class_labels` for SongUNet
    unet_class_labels = None
    if unet_label_dim > 0:
        if cond_input is None:
            # If model expects labels but none are provided, error or use zeros
            # For classifier-free guidance, this might be a zeroed tensor
            # For now, assume cond_input is always provided if unet_label_dim > 0
            raise ValueError(
                "cond_input is None but unet_label_dim > 0."
            )
        # Assuming cond_input are integer labels (B,)
        # Convert to one-hot if model expects label_dim > 0
        # this should be in the model dtype as is the input of a linear layer
        unet_class_labels = F.one_hot(
            cond_input.long(), num_classes=unet_label_dim
        ).to(dtype)
    # If unet_label_dim is 0, unet_class_labels remains None (unconditional)

    # Predict velocity using SongUNet
    # Input to SongUNet: x_scaled = xt * c_in_val
    # Pass `augment_labels=None` as we are not handling them here.
    v_pred = unet_model(
        x=xt * c_in_val,
        noise_labels=unet_noise_labels,
        class_labels=unet_class_labels,
        augment_labels=None # Assuming not used for this FM fine-tuning
    )
    # print(f"Norm v_pred: {torch.norm(v_pred).item():.4f}, Norm v_true: {torch.norm(v_true).item():.4f}")
    # print(f"Mean Abs v_pred: {torch.mean(torch.abs(v_pred)).item():.4f}, Mean Abs v_true: {torch.mean(torch.abs(v_true)).item():.4f}")

    loss = F.mse_loss(v_pred, v_true, reduction="mean")
    return loss

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
    t_steps = torch.linspace(0, 1, num_inference_steps + 1, device=device)
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
        for i in tqdm(range(num_inference_steps), desc="Sampling (EDM-FM)"):
            t_current = t_steps[i]
            dt = t_steps[i+1] - t_current

            # Replicate t_current for batch
            t_batch = torch.full((num_samples,), t_current.item(), device=device)

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

# --- Improved Learning Rate Grouping for SongUNet ---
def create_songunet_lr_groups(
    song_unet_model: nn.Module,
    base_lr: float,
    img_resolution: int, # e.g., 32 for CIFAR-10
    output_lr_multiplier: float = 5.0,
    time_embed_lr_multiplier: float = 3.0,
    backbone_lr_multiplier: float = 0.1
) -> List[Dict]:
    """
    Creates parameter groups with different LRs for SongUNet.
    Targets specific layers/modules within SongUNet.

    Args:
        song_unet_model: The SongUNet model instance.
        base_lr: The base learning rate.
        img_resolution: Image resolution to identify final output layers.
        output_lr_multiplier: LR multiplier for final output layers.
        time_embed_lr_multiplier: LR multiplier for time embedding layers.
        backbone_lr_multiplier: LR multiplier for the rest of the backbone.
    """
    param_groups = []

    # Define specific names for output layers based on SongUNet structure
    output_param_names_exact = [
        f"dec.{img_resolution}x{img_resolution}_aux_norm.weight",
        f"dec.{img_resolution}x{img_resolution}_aux_norm.bias",
        f"dec.{img_resolution}x{img_resolution}_aux_conv.weight",
        f"dec.{img_resolution}x{img_resolution}_aux_conv.bias",
    ]

    # Time embedding layers in SongUNet are typically named 'map_*'
    # and are direct attributes of the SongUNet model.
    time_embed_patterns_regex = [
        r"^map_noise\.", r"^map_label\.", r"^map_augment\.",
        r"^map_layer0\.", r"^map_layer1\."
    ]

    output_params, time_embed_params, backbone_params = [], [], []

    print("Categorizing SongUNet parameters for learning rate groups...")
    for name, param in song_unet_model.named_parameters():
        if not param.requires_grad:
            continue

        is_output = name in output_param_names_exact
        is_time_embed = any(
            re.match(p, name) for p in time_embed_patterns_regex
        )

        if is_output:
            output_params.append(param)
        elif is_time_embed:
            time_embed_params.append(param)
        else:
            backbone_params.append(param)

    if output_params:
        param_groups.append({
            'params': output_params,
            'lr': base_lr * output_lr_multiplier, # Initial LR
            'name': 'output_layers',
            'lr_multiplier': output_lr_multiplier # Store multiplier
        })
        print(f"  Output layers ({len(output_params)} params): "
              f"LR={base_lr * output_lr_multiplier:.2e} "
              f"(multiplier: {output_lr_multiplier}x)")
    if time_embed_params:
        param_groups.append({
            'params': time_embed_params,
            'lr': base_lr * time_embed_lr_multiplier,
            'name': 'time_embed_layers',
            'lr_multiplier': time_embed_lr_multiplier
        })
        print(f"  Time embed layers ({len(time_embed_params)} params): "
              f"LR={base_lr * time_embed_lr_multiplier:.2e} "
              f"(multiplier: {time_embed_lr_multiplier}x)")
    if backbone_params:
        param_groups.append({
            'params': backbone_params,
            'lr': base_lr * backbone_lr_multiplier,
            'name': 'backbone_layers',
            'lr_multiplier': backbone_lr_multiplier
        })
        print(f"  Backbone layers ({len(backbone_params)} params): "
              f"LR={base_lr * backbone_lr_multiplier:.2e} "
              f"(multiplier: {backbone_lr_multiplier}x)")

    if not param_groups and list(song_unet_model.parameters()):
        # Fallback: create a default group if others are empty
        all_params = [
            p for p in song_unet_model.parameters() if p.requires_grad
        ]
        if all_params:
            param_groups.append({
                'params': all_params,
                'lr': base_lr,
                'name': 'default_group',
                'lr_multiplier': 1.0
            })
            print("Warning: No specific LR groups made or all specific groups "
                  "were empty. Using base_lr for all parameters in a "
                  "default group.")
    elif not any(pg['name'] == 'backbone_layers' for pg in param_groups) and \
         not any(pg['name'] == 'default_group' for pg in param_groups):
        # Ensure any remaining params are caught if logic missed them
        all_categorized_params = set()
        for group in param_groups:
            for p in group['params']:
                all_categorized_params.add(p)

        remaining_params = [
            p for p in song_unet_model.parameters()
            if p.requires_grad and p not in all_categorized_params
        ]
        if remaining_params:
            param_groups.append({
                'params': remaining_params,
                'lr': base_lr * backbone_lr_multiplier, # Default to backbone
                'name': 'remaining_backbone_layers',
                'lr_multiplier': backbone_lr_multiplier
            })
            print(f"  Remaining backbone layers "
                  f"({len(remaining_params)} params): "
                  f"LR={base_lr * backbone_lr_multiplier:.2e} "
                  f"(multiplier: {backbone_lr_multiplier}x)")


    return param_groups


def save_images_for_fid(
    images_tensor: torch.Tensor, # (B, C, H, W), range [-1, 1] float
    output_dir: str,
    start_idx: int = 0,
    file_prefix: str = "sample"
):
    """Saves a batch of image tensors as PNG files for FID calculation."""
    os.makedirs(output_dir, exist_ok=True)
    # Denormalize from [-1, 1] to [0, 1], then to [0, 255] uint8
    images_tensor = (images_tensor + 1) / 2.0
    images_tensor = images_tensor.mul(255).add_(0.5).clamp_(0, 255)
    images_tensor = images_tensor.permute(0, 2, 3, 1).to('cpu', torch.uint8)
    images_np = images_tensor.numpy()

    for i, img_np_chw in enumerate(images_np):
        # img_np_chw is (H, W, C)
        if img_np_chw.shape[-1] == 1: # Grayscale
            img_pil = Image.fromarray(img_np_chw[..., 0], mode='L')
        else: # RGB
            img_pil = Image.fromarray(img_np_chw, mode='RGB')
        
        filename = f"{file_prefix}_{start_idx + i:06d}.png"
        img_pil.save(os.path.join(output_dir, filename))

def calculate_fid_stats_for_path(
    image_path: str,
    inception_model: torch.nn.Module, # Now directly takes the model
    num_expected: int,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 2,
    prefetch_factor: int = 2,
    seed: int = 0, 
):
    """
    Calculates Inception statistics (mu, sigma) for images in a directory.
    Requires edm_dataset to be available.
    """
    if edm_dataset is None:
        raise ImportError(
            "training.dataset (from EDM) not available. "
            "Cannot calculate FID stats for path."
        )
    if inception_model is None:
        raise ValueError("Inception model not provided to "
                         "calculate_fid_stats_for_path.")

    feature_dim = 2048 
    detector_kwargs = dict(return_features=True)

    print(f'Loading images for FID from "{image_path}"...')
    dataset_obj = edm_dataset.ImageFolderDataset(
        path=image_path,
        max_size=num_expected,
        random_seed=seed,
        use_labels=False, 
        resolution=None 
    )
    if len(dataset_obj) < num_expected:
        print(f"Warning: Found {len(dataset_obj)} images, "
              f"but expected {num_expected}.")
    if len(dataset_obj) < 2:
        raise ValueError(f"Need at least 2 images to compute FID stats, "
                         f"found {len(dataset_obj)}")

    data_loader = torch.utils.data.DataLoader(
        dataset_obj,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        shuffle=False, # Order doesn't matter for stats
        drop_last=False
    )

    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim],
                        dtype=torch.float64, device=device)
    
    inception_model.eval() # Ensure model is in eval mode

    print(f'Calculating Inception stats for {len(dataset_obj)} images...')
    total_processed = 0
    for images, _labels in tqdm(data_loader, desc="FID Stats"):
        if images.shape[0] == 0:
            continue
        
        # Ensure images are 3-channel (repeat grayscale if necessary)
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        
        # Images from edm_dataset.ImageFolderDataset are already [-1, 1]
        features = inception_model(
            images.to(device), **detector_kwargs
        ).to(torch.float64)
        
        mu += features.sum(0)
        sigma += features.T @ features
        total_processed += images.shape[0]

    mu /= total_processed
    sigma -= torch.outer(mu, mu) * total_processed
    sigma /= (total_processed - 1) if total_processed > 1 else 1
    
    return mu.cpu().numpy(), sigma.cpu().numpy()

def calculate_fid_score(
    mu_gen: np.ndarray,
    sigma_gen: np.ndarray,
    ref_stats_path: str
) -> float:
    """Calculates FID score given generated and reference stats."""
    print(f'Loading reference FID stats from "{ref_stats_path}"...')
    try:
        with np.load(ref_stats_path) as ref_data:
            mu_ref = ref_data['mu']
            sigma_ref = ref_data['sigma']
    except Exception as e:
        print(f"Error loading reference FID stats: {e}")
        raise

    m = np.square(mu_gen - mu_ref).sum()
    # Note: scipy.linalg.sqrtm can be slow and numerically unstable.
    # Consider adding epsilon to diagonal of sigma for stability if issues arise.
    s, _ = scipy.linalg.sqrtm(
        sigma_gen @ sigma_ref, disp=False
    ) 
    
    # If sqrtm returns complex numbers, take the real part
    if np.iscomplexobj(s):
        s = np.real(s)

    fid = m + np.trace(sigma_gen + sigma_ref - 2 * s)
    return float(np.real(fid))


def train_flow_matching_edm_with_songunet(
    # model is the raw SongUNet instance
    model: torch.nn.Module,
    train_loader, # Assumes data loader yields (images, raw_cond_input)
    test_loader,
    epochs: int = 10,
    patience: int = 5,
    eval_interval: int = 100, # Adjusted for potentially larger datasets
    lr: float = 1e-4, # Common starting LR for FM fine-tuning
    wd: float = 1e-6,
    device: str = "cuda",
    seed: int = None,
    img_shape: tuple = (3, 512, 512), # As per project description
    edm_params: dict = None,
    # For generation at epoch end:
    num_gen_samples_epoch_end: int = 10,
    num_inference_steps_epoch_end: int = 50,
    dtype=torch.bfloat16,
    # W&B related arguments
    wandb_project: str = "edm_fm_finetuning",
    wandb_run_name: str = None,
    wandb_entity: str = None,
    log_config: dict = None, # For logging hyperparameters
    inception_model: torch.nn.Module = None, # Pass pre-loaded model
    fid_eval_epochs: int = 0, # Default to 0 (disabled)
    fid_num_samples: int = 10000, 
    fid_gen_batch_size: int = 64, 
    fid_inception_batch_size: int = 64, 
    fid_ref_path: str = None, 
    fid_num_workers: int = 2   
):
    if edm_params is None:
        edm_params = {
            'sigma_min': 0.002, 'sigma_max': 80.0, 'sigma_data': 0.5
        }

    # Initialize W&B
    if log_config is None: # Basic config if not provided
        log_config = {
            "epochs": epochs, "learning_rate": lr, "weight_decay": wd,
            "batch_size": train_loader.batch_size if hasattr(train_loader, 'batch_size') else 'N/A',
            "eval_interval": eval_interval, "patience": patience,
            "seed": seed, "img_shape": img_shape,
            "num_gen_samples": num_gen_samples_epoch_end,
            "num_inference_steps": num_inference_steps_epoch_end,
            "edm_params": edm_params,
            "fid_eval_epochs": fid_eval_epochs,
            "fid_num_samples": fid_num_samples,
            "fid_gen_batch_size": fid_gen_batch_size,
            "fid_inception_batch_size": fid_inception_batch_size,
            "fid_ref_path": fid_ref_path,
            "fid_num_workers": fid_num_workers
        }
    logging_utils.init_wandb(
        project_name=wandb_project,
        run_name=wandb_run_name,
        config=log_config,
        entity=wandb_entity
    )

    # Infer label_dim from the SongUNet model
    # SongUNet stores label_dim directly or via self.map_label
    model_label_dim = 0
    if hasattr(model, 'map_label') and model.map_label is not None:
        model_label_dim = model.map_label.in_features
    elif hasattr(model, 'label_dim'): # If stored directly
        model_label_dim = model.label_dim
    print(f"Using model_label_dim: {model_label_dim}")
    if logging_utils.is_wandb_initialized():
        logging_utils.log_metrics({"model/label_dim": model_label_dim}, commit=False)

    torch.cuda.empty_cache()
    img_resolution = img_shape[1] # Assuming H=W

    # 2. Setup optimizer with grouped learning rates
    # Calculate total_train_steps = epochs * len(train_loader)
    param_groups = create_songunet_lr_groups(
        model,
        base_lr=lr, # Your base LR
        img_resolution=img_resolution,
        # You can tune these multipliers:
        output_lr_multiplier=3.0,
        time_embed_lr_multiplier=2.0,
        backbone_lr_multiplier=1.0 # Maybe slightly higher for backbone
    )


    if not param_groups: # Fallback if no groups were created
        print("Warning: No parameter groups created. Using all parameters with base LR.")
        optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd,
            fused=(device=="cuda" and torch.cuda.is_available())
        )
    else:
        optimizer = optim.AdamW(
            param_groups, weight_decay=wd, # LR is set in groups
            fused=(device=="cuda" and torch.cuda.is_available())
        )

    if hasattr(train_loader, '__len__') and len(train_loader) > 0:
        t_max_steps = epochs * len(train_loader)
    else:
        print("Warning: train_loader length unknown, using estimated T_max.")
        estimated_steps_per_epoch = 500000 // (train_loader.batch_size if train_loader.batch_size else 16)
        t_max_steps = epochs * estimated_steps_per_epoch
        if t_max_steps == 0: t_max_steps = 10000
    
    warmup_steps = round(0.02 * t_max_steps) if t_max_steps > 0 else 0
    # Ensure warmup_steps is not greater than t_max_steps
    warmup_steps = min(warmup_steps, t_max_steps -1) if t_max_steps > 0 else 0
    
    print(f"Total training steps: {t_max_steps}, Warmup steps: {warmup_steps}")

    if t_max_steps > 0 and warmup_steps < t_max_steps :
        scheduler_1 = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_steps if warmup_steps > 0 else 1 # total_iters must be > 0
        )
        # Ensure T_max for CosineAnnealingLR is at least 1
        cosine_t_max = t_max_steps - warmup_steps
        if cosine_t_max <= 0: cosine_t_max = 1

        scheduler_2 = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_t_max, eta_min=lr * 0.01 # Adjusted eta_min
        )
        if warmup_steps > 0:
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer, [scheduler_1, scheduler_2], milestones=[warmup_steps]
            )
        else: # No warmup, only cosine annealing
            scheduler = scheduler_2
    else: # Not enough steps for a scheduler, or no steps
        scheduler = None # Or a dummy scheduler
        print("Warning: Not enough training steps for LR scheduling or t_max_steps is 0.")

    generator = torch.Generator(device=device)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        generator.manual_seed(seed)
    
    model.to(device)

    best_eval_loss = float('inf')
    current_patience = 0
    global_step = 0

    try:
        print(f"Starting Flow Matching training for SongUNet (EDM pre-trained)...")
        for epoch in range(epochs):
            model.train()
            epoch_loss_accum = 0.0
            num_batches_epoch = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

            for batch_idx, batch in enumerate(progress_bar):
                optimizer.zero_grad()

                x1_batch = batch[0].to(dtype, device)
                cond_input_batch = batch[1].to(dtype, device) if len(batch) > 1 else None

                optimizer.zero_grad()

                loss = flow_matching_step(
                    unet_model=model,
                    x1=x1_batch,
                    cond_input=cond_input_batch,
                    generator=generator,
                    input_perturbation=True, # Often good for FM
                    edm_params=edm_params,
                    unet_label_dim=model_label_dim,
                    dtype=dtype
                )

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected at step {global_step}. Skipping batch.")
                    # Optionally, try to recover or stop training
                    # For now, just skip optimizer step and backward
                    if logging_utils.is_wandb_initialized():
                        logging_utils.log_metrics(
                            {"train/skipped_batch_due_to_nan_inf_loss": 1},
                            step=global_step
                        )
                    continue # Skip to next batch

                loss.backward()

                # Track gradient norms
                # grads = [
                #     param.grad.detach().flatten()
                #     for param in model.parameters()
                #     if param.grad is not None
                # ]
                # norm = torch.cat(grads).norm()

                # returns the grad norm like the code above
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler:
                    scheduler.step()

                epoch_loss_accum += loss.item()
                num_batches_epoch += 1
                global_step += 1

                if logging_utils.is_wandb_initialized():
                    log_payload = {
                        "train/loss_step": loss.item(),
                        "train/grad_norm_step": norm.item(),
                        "learning_rate": optimizer.param_groups[0]['lr']
                    }
                    # Log LR for each group if multiple groups exist
                    for i, pg in enumerate(optimizer.param_groups):
                        log_payload[f"lr_group_{pg.get('name', i)}"] = pg['lr']

                    logging_utils.log_metrics(log_payload, step=global_step)

                # Evaluation and logging
                if (batch_idx + 1) % eval_interval == 0 or \
                   (batch_idx + 1) == len(train_loader):
                    avg_train_loss_interval = epoch_loss_accum / num_batches_epoch
                    
                    # Validation
                    model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
                            x1_batch = batch[0].to(dtype, device)
                            cond_input_batch = batch[1].to(dtype, device) if len(batch) > 1 else None
                            loss = flow_matching_step(
                                unet_model=model,
                                x1=x1_batch,
                                cond_input=cond_input_batch,
                                generator=None, # No specific generator for eval
                                input_perturbation=False, # No perturbation for eval
                                edm_params=edm_params,
                                unet_label_dim=model_label_dim,
                                dtype=dtype
                            )
                            val_loss += loss.item()

                    val_loss = val_loss / len(test_loader)

                    print(f"\nEpoch {epoch+1}, Step {global_step}: "
                          f"Train Loss: {avg_train_loss_interval:.4f}, "
                          f"Val Loss: {val_loss:.4f}")

                    if logging_utils.is_wandb_initialized():
                        logging_utils.log_metrics({
                            "eval/avg_val_loss": val_loss,
                            "train/avg_loss_interval": avg_train_loss_interval,
                            "epoch": epoch + (batch_idx + 1) / len(train_loader)
                        }, step=global_step)

                    # Checkpointing and Early Stopping
                    if val_loss < best_eval:
                        best_eval = val_loss
                        current_patience = 0
                        save_path = os.path.join(
                            logging_utils._wandb_run.dir if logging_utils.is_wandb_initialized() and hasattr(logging_utils._wandb_run, 'dir') else ".",
                            "edm_fm_best_model.pt"
                        )
                        torch.save(model.state_dict(), save_path)
                        print(f"  -> New best val loss: {best_eval_loss:.4f}. Model saved to {save_path}")
                        if logging_utils.is_wandb_initialized():
                            wandb.save(save_path, base_path=os.path.dirname(save_path)) # Save to W&B artifacts
                    else:
                        current_patience += 1
                        if current_patience >= patience:
                            print(f"Early stopping triggered after {patience} evaluations without improvement.")
                            if logging_utils.is_wandb_initialized():
                                logging_utils.log_metrics({"training_control/early_stopped": True}, step=global_step)
                            raise StopIteration # Use custom exception or break
                    model.train()
            
            # --- End of Epoch ---
            avg_epoch_loss = epoch_loss_accum / num_batches_epoch if num_batches_epoch > 0 else float('inf')
            print(f"End of Epoch {epoch+1}/{epochs}: Avg Train Loss: {avg_epoch_loss:.4f}")
            if logging_utils.is_wandb_initialized():
                logging_utils.log_metrics({
                    "train/avg_epoch_loss": avg_epoch_loss,
                    "epoch": epoch + 1
                }, step=global_step)

            # Generate and log samples at the end of each epoch
            if num_gen_samples_epoch_end > 0:
                print("Generating samples for W&B logging...")
                model.eval()
                # Generate for a few fixed labels if conditional, or unconditionally
                gen_labels = None
                if model_label_dim > 0:
                    # Ensure labels are within the valid range [0, model_label_dim - 1]
                    num_labels_to_gen = min(model_label_dim, num_gen_samples_epoch_end)
                    gen_labels = torch.arange(num_labels_to_gen, device=device) \
                                     if num_labels_to_gen > 0 else None
                
                # If gen_labels is still None (e.g. unconditional or model_label_dim is 0)
                # and we still want to generate samples, num_gen_samples_epoch_end will be used.
                actual_num_samples_to_gen = num_gen_samples_epoch_end
                if gen_labels is not None:
                    actual_num_samples_to_gen = len(gen_labels)


                if actual_num_samples_to_gen > 0:
                    images, traj = generate_samples_edm_fm(
                        model, num_samples=actual_num_samples_to_gen,
                        cond_input=gen_labels, device=device,
                        num_inference_steps=num_inference_steps_epoch_end,
                        img_shape=img_shape, edm_params=edm_params,
                        seed=seed + epoch if seed is not None else None, # Vary seed per epoch
                        unet_label_dim=model_label_dim,
                        dtype=dtype
                    )
                    
                    # Log the final generated image grid
                    # plot_generation_steps now returns the filename
                    # Use a unique filename for each epoch to avoid overwriting in W&B artifacts
                    timestamp = datetime.now().strftime(r'%Y%m%d-%H%M%S')
                    trajectory_plot_filename = f"trajectory_epoch_{epoch+1}_{timestamp}.png"
                    
                    # Save plot locally (useful for non-W&B runs or local checks)
                    # The plot_generation_steps function already saves the file.
                    # We need to ensure the path is accessible for wandb.
                    # If wandb run dir is available, save there.
                    save_dir = "."
                    if logging_utils.is_wandb_initialized() and hasattr(logging_utils._wandb_run, 'dir'):
                        save_dir = logging_utils._wandb_run.dir

                    full_plot_path = os.path.join(save_dir, trajectory_plot_filename)

                    saved_plot_path = plot_generation_steps(
                        traj, num_images_to_show=min(8, actual_num_samples_to_gen), # Show up to 8 images in the plot
                        filename=full_plot_path,
                        title=f"FM Samples Epoch {epoch+1}"
                    )

                    if saved_plot_path and logging_utils.is_wandb_initialized():
                        logging_utils.log_image(
                            "epoch_samples/generation_trajectory",
                            saved_plot_path,
                            caption=f"Trajectory at Epoch {epoch+1}",
                            step=global_step
                        )
                        # Log individual final images as well for easier viewing
                        final_images_to_log = min(images.size(0), 16) # Log up to 16 final images
                        for i in range(final_images_to_log):
                            img_to_log = (images[i].permute(1,2,0).numpy() + 1) / 2 # Denorm HWC
                            img_to_log = np.clip(img_to_log, 0, 1)
                            logging_utils.log_image(
                                f"epoch_samples/final_sample_{i}",
                                img_to_log,
                                caption=f"Final Sample {i} Epoch {epoch+1} Label: {gen_labels[i].item() if gen_labels is not None and i < len(gen_labels) else 'N/A'}",
                                step=global_step
                            )
            
                perform_fid = (
                    fid_ref_path and 
                    inception_model is not None and # Check if model was passed
                    edm_dataset is not None and # Check if edm_dataset is available
                    (epoch + 1) % fid_eval_epochs == 0 and 
                    fid_num_samples > 0 and
                    fid_eval_epochs > 0 # Ensure fid_eval_epochs is positive
                )

                if perform_fid:
                    print(f"Epoch {epoch+1}: Starting FID calculation with "
                        f"{fid_num_samples} samples...")
                    model.eval()
                    fid_temp_dir = tempfile.mkdtemp()
                    print(f"  Saving generated images for FID to: {fid_temp_dir}")
                    
                    generated_count = 0
                    try:
                        fid_gen_labels = None
                        if model_label_dim > 0:
                            pass # Labels handled per batch

                        for i in tqdm(range(0, fid_num_samples, fid_gen_batch_size), # noqa
                                    desc="Generating FID Samples"):
                            current_batch_size = min(
                                fid_gen_batch_size,
                                fid_num_samples - generated_count
                            )
                            if current_batch_size <= 0:
                                break

                            batch_gen_labels = None
                            if model_label_dim > 0:
                                batch_gen_labels = torch.randint(
                                    0, model_label_dim, (current_batch_size,),
                                    device=device
                                )
                            
                            fid_seed = seed + epoch + i if seed is not None else None
                            batch_images, _ = generate_samples_edm_fm(
                                model,
                                num_samples=current_batch_size,
                                cond_input=batch_gen_labels,
                                device=device,
                                num_inference_steps=num_inference_steps_epoch_end,
                                img_shape=img_shape,
                                edm_params=edm_params,
                                seed=fid_seed,
                                unet_label_dim=model_label_dim,
                                dtype=dtype 
                            )
                            save_images_for_fid(
                                batch_images, fid_temp_dir, 
                                start_idx=generated_count
                            )
                            generated_count += current_batch_size
                        
                        print(f"  Generated {generated_count} samples for FID.")
                        if generated_count >= 2: 
                            mu_gen, sigma_gen = calculate_fid_stats_for_path(
                                image_path=fid_temp_dir,
                                inception_model=inception_model, # Use passed model
                                num_expected=generated_count, 
                                device=torch.device(device),
                                batch_size=fid_inception_batch_size,
                                num_workers=fid_num_workers
                            )
                            fid_score = calculate_fid_score(
                                mu_gen, sigma_gen, fid_ref_path
                            )
                            print(f"  FID Score (Epoch {epoch+1}): {fid_score:.4f}")
                            if logging_utils.is_wandb_initialized():
                                logging_utils.log_metrics(
                                    {"eval/fid_score": fid_score},
                                    step=global_step
                                )
                        else:
                            print("  Not enough samples generated for FID.")

                    except ImportError as e_import:
                        print(f"ImportError during FID: {e_import}. "
                            "FID calculation skipped. Ensure EDM toolkit "
                            "components are available.")
                        if logging_utils.is_wandb_initialized():
                            logging_utils.log_metrics(
                                {"eval/fid_error_import": 1}, step=global_step
                            )
                    except Exception as e:
                        print(f"Error during FID calculation: {e}")
                        import traceback
                        traceback.print_exc()
                        if logging_utils.is_wandb_initialized():
                            logging_utils.log_metrics(
                                {"eval/fid_error_runtime": 1}, step=global_step
                            )
                    finally:
                        print(f"  Cleaning up FID temp directory: {fid_temp_dir}")
                        shutil.rmtree(fid_temp_dir)
                        model.train() 
                elif (epoch + 1) % fid_eval_epochs == 0 and fid_eval_epochs > 0:
                    # Log why FID was skipped if it was supposed to run
                    reason = []
                    if not fid_ref_path: reason.append("fid_ref_path not set")
                    if inception_model is None: reason.append("Inception model not loaded") # noqa
                    if edm_dataset is None: reason.append("edm_dataset not available") # noqa
                    if fid_num_samples <= 0: reason.append("fid_num_samples <= 0")
                    print(f"FID calculation skipped for epoch {epoch+1}. "
                        f"Reasons: {', '.join(reason) if reason else 'Unknown'}.")


                model.train() # Back to training mode

            # Save epoch checkpoint (optional, can be frequent)
            epoch_save_path = os.path.join(
                logging_utils._wandb_run.dir if logging_utils.is_wandb_initialized() and hasattr(logging_utils._wandb_run, 'dir') else ".",
                f"edm_fm_epoch_{epoch+1}.pt"
            )
            torch.save(model.state_dict(), epoch_save_path)
            if logging_utils.is_wandb_initialized():
                 wandb.save(epoch_save_path, base_path=os.path.dirname(epoch_save_path), policy="end") # Save at end of run or if "live"

        print("Training finished normally.")


    except StopIteration: # Handles early stopping
        print("Early stopping condition met. Training halted.")
    except KeyboardInterrupt:
        print("Training interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        if logging_utils.is_wandb_initialized():
            logging_utils.log_metrics({"training_control/error_occurred": 1}, step=global_step if 'global_step' in locals() else None)
            # You might want to log the error message to W&B as well
            # _wandb_run.summary["error_message"] = str(e)
    finally:
        if logging_utils.is_wandb_initialized():
            # Save final model if not early stopped or if desired
            final_save_path = os.path.join(
                logging_utils._wandb_run.dir if hasattr(logging_utils._wandb_run, 'dir') else ".",
                "edm_fm_final_model.pt"
            )
            torch.save(model.state_dict(), final_save_path)
            wandb.save(final_save_path, base_path=os.path.dirname(final_save_path))
            print(f"Final model saved to {final_save_path} and W&B.")
            logging_utils.finish_wandb()

    # Return model and any tracked local metrics if needed outside W&B
    return model # Placeholder for local metrics if you re-enable them
