import torch
from PIL import Image
import os
import numpy as np
import scipy.linalg
try:
    # For ImageFolderDataset used in FID calculation
    from training import dataset as edm_dataset
except ImportError:
    print("Warning: training.dataset (from EDM) not found. "
          "FID calculation will be disabled if it relies on this.")
    edm_dataset = None # Set to None if not found
from training.plot_utils import generate_samples_edm_fm
from tqdm.auto import tqdm
import tempfile
import shutil

def save_images_for_fid(
    images_tensor: torch.Tensor, # (B, C, H, W), range [-1, 1] float
    output_dir: str,
    start_idx: int = 0,
    file_prefix: str = "sample"
):
    """Saves a batch of image tensors as PNG files for FID calculation."""
    os.makedirs(output_dir, exist_ok=True)
    # Denormalize from [-1, 1] to [0, 1], then to [0, 255] uint8
    images_tensor = (images_tensor.float() + 1) / 2.0
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
    
    return mu.float().cpu().numpy(), sigma.float().cpu().numpy()

def calculate_fid_score(
    mu_gen: np.ndarray,
    sigma_gen: np.ndarray,
    ref_stats: str
) -> float:
    """Calculates FID score given generated and reference stats."""
    mu_ref = ref_stats['mu']
    sigma_ref = ref_stats['sigma']

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

def evaluate_fid(
    model,
    epoch,
    global_step,
    fid_ref,
    inception_model,
    logging_utils,
    # FID configuration parameters
    fid_num_samples,
    fid_gen_batch_size,
    fid_inception_batch_size,
    fid_num_workers,
    # Model and generation parameters
    model_label_dim,
    device,
    seed,
    num_inference_steps_epoch_end,
    img_shape,
    edm_params,
    dtype
):
    """
    Performs FID evaluation for the given model and logs the results.
    
    Args:
        model: The generative model to evaluate
        epoch: Current training epoch (0-indexed)
        global_step: Current global training step
        fid_ref: Reference FID statistics dictionary
        inception_model: Pre-loaded Inception model for feature extraction
        logging_utils: Logging utility object with wandb integration
        generate_samples_edm_fm: Function to generate samples using the model
        fid_num_samples: Number of samples to generate for FID calculation
        fid_gen_batch_size: Batch size for sample generation
        fid_inception_batch_size: Batch size for Inception feature extraction
        fid_num_workers: Number of workers for data loading
        model_label_dim: Dimension of model labels (0 if unconditional)
        device: Device to run computations on
        seed: Random seed for reproducible generation
        num_inference_steps_epoch_end: Number of inference steps for generation
        img_shape: Shape of generated images
        edm_params: EDM-specific parameters
        dtype: Data type for computations
    
    Returns:
        None (results are logged and printed)
    """
    print(f"Epoch {epoch+1}: Starting FID calculation with "
          f"{fid_num_samples} samples...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create temporary directory for generated images
    fid_temp_dir = tempfile.mkdtemp()
    print(f"  Saving generated images for FID to: {fid_temp_dir}")
    
    generated_count = 0
    
    try:
        # Generate samples in batches
        for i in tqdm(range(0, fid_num_samples, fid_gen_batch_size),
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
            
            # Generate deterministic samples using seed
            fid_seed = seed + epoch + i if seed is not None else None
            
            # Generate batch of images
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
            
            # Save generated images to temporary directory
            save_images_for_fid(
                batch_images, fid_temp_dir, 
                start_idx=generated_count
            )
            generated_count += current_batch_size
        
        print(f"  Generated {generated_count} samples for FID.")
        
        # Calculate FID score if we have enough samples
        if generated_count >= 2: 
            # Calculate statistics for generated images
            mu_gen, sigma_gen = calculate_fid_stats_for_path(
                image_path=fid_temp_dir,
                inception_model=inception_model,
                num_expected=generated_count, 
                device=torch.device(device),
                batch_size=fid_inception_batch_size,
                num_workers=fid_num_workers
            )
            
            # Compute FID score against reference statistics
            fid_score = calculate_fid_score(
                mu_gen, sigma_gen, fid_ref
            )
            
            print(f"  FID Score (Epoch {epoch+1}): {fid_score:.4f}")
            
            # Log FID score if wandb is available
            if logging_utils.is_wandb_initialized():
                logging_utils.log_metrics(
                    {"eval/fid_score": fid_score},
                    step=global_step,
                    commit=False
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
        # Clean up temporary directory and restore model to training mode
        print(f"  Cleaning up FID temp directory: {fid_temp_dir}")
        shutil.rmtree(fid_temp_dir)
        model.train()