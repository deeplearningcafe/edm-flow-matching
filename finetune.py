# download model
import dnnlib
import pickle
from datetime import datetime
import torch
import random
import numpy as np
try:
    from training.dataset import (
        load_cifar10_torch,
        create_aligned_stl10_dataloader,
        create_aligned_stl10_test_dataloader
    )
    from training.fm import train_flow_matching_edm_with_songunet
    from training.networks import Unet, UnetConfig, weights_init
except ImportError:
    # Fallback if they are in the same directory or top-level
    from training.dataset import (
        load_cifar10_torch,
        create_aligned_stl10_dataloader,
        create_aligned_stl10_test_dataloader
    )
    from training.fm import train_flow_matching_edm_with_songunet

torch.manual_seed(46)
random.seed(46)
np.random.seed(46)

def load_trainable_model(network_pkl, device, dtype=torch.float32):
    # Ensure dtype is applied when loading or after
    print(f"Loading model from {network_pkl} to device {device} with dtype {dtype}")
    try:
        with dnnlib.util.open_url(network_pkl, cache=True, cache_dir='.') as f:
            # Load to CPU first, then move and cast to avoid VRAM issues with large models
            net = pickle.load(f)['ema'].to("cpu")
        
        # Apply dtype conversion before moving to target device if it's CUDA
        # This is safer for mixed precision components within the model
        if dtype != torch.float32: # Default pickle load is float32
            net = net.to(dtype=dtype)
        net = net.to(device)

    except FileNotFoundError:
        print(f"ERROR: Model PKL file not found at {network_pkl}.")
        raise
    except Exception as e:
        print(f"ERROR: Could not load model: {e}")
        raise
    
    unet = net.model # Assuming net always has a .model attribute
    
    # Enable gradients for all parameters in the U-Net part
    for param in unet.parameters():
        param.requires_grad = True
    unet.train() # Set to training mode

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Trainable parameters in unet (net.model): {count_parameters(unet)}")
    return unet # Return the U-Net part directly

def load_inception_model(
    device: torch.device,
    detector_url: str = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl',
    ref_url: str = 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz'
):
    """Loads the Inception-v3 model for FID calculation."""
    if dnnlib is None or pickle is None:
        print("Warning: dnnlib or pickle module not available. "
              "Cannot load Inception model for FID.")
        return None
    
    print(f'Loading Inception-v3 model from {detector_url}...')
    try:
        with dnnlib.util.open_url(detector_url, verbose=True) as f:
            # Load to CPU first, then move to device
            inception_model = pickle.load(f).to("cpu")
        inception_model = inception_model.to(device)
        inception_model.eval() # Ensure it's in eval mode
        print('Inception-v3 model loaded successfully.')
    except Exception as e:
        print(f"Error loading Inception-v3 model: {e}. "
              "FID calculation will be disabled.")
        return None

    try:
        with dnnlib.util.open_url(ref_url) as f:
            ref = dict(np.load(f))
        print("Reference loaded successfully.")
        return inception_model, ref

    except Exception as e:
        print(f"Error loading Ref : {e}. "
              "FID calculation will be disabled.")
        return inception_model, None

def unet_model_custom():
    config = UnetConfig() # Uses defaults: 4 blocks, 64 init channels, etc.

    # Instantiate custom Unet model
    unet_model = Unet(config)
    unet_model.apply(weights_init)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Trainable parameters in unet (net.model): {count_parameters(unet_model)}")
    return unet_model

def train(
        network_pkl='https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl',
        dataset_name="cifar10",
        batch_size=64,
        learning_rate = 6e-5,
        epochs = 10,
        eval_interval = 100,
        patience = 4,
        num_workers=2,
        wd = 0.02,
        num_gen_samples_epoch_end=10, # Generate 10 samples
        num_inference_steps_epoch_end=50,
        dtype=torch.bfloat16,
        # W&B arguments
        wandb_project_name="edm_fm_finetune_cifar10_shift",
        wandb_run_name_prefix="cifar10-fm", # Will append timestamp
        wandb_entity=None, # Your W&B username or team
        seed=46,
        # FID specific parameters to be passed to training function
        fid_eval_epochs: int = 0, # Default to 0 (disabled)
        fid_num_samples: int = 10000,
        fid_gen_batch_size: int = 64,
        fid_inception_batch_size: int = 64,
        fid_ref_path: str = None, # e.g., "fid-refs/cifar10-32x32.npz"
        fid_num_workers: int = 2,
        use_custom_unet=False,
        hf_repo:str = "",
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed) # Ensure all seeds are set

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"CUDA version: {torch.version.cuda}")
        capability = torch.cuda.get_device_capability()
        # BF16 support generally on Ampere (8.0+) and Hopper (9.0+)
        # Volta (7.0) supports FP16 well, limited BF16.
        if dtype == torch.bfloat16 and capability[0] < 8:
            print(f"Warning: bfloat16 specified but GPU capability "
                  f"({capability[0]}.{capability[1]}) may not fully support it. "
                  f"Consider float16 or float32.")
        # Set matmul precision (good for Ampere+)
        if capability[0] >= 7:
            torch.set_float32_matmul_precision("high")
            print("Using high precision for float32 matmul (tensor cores).")
        else:
            print("Tensor cores for float32 matmul not optimally supported "
                  "or GPU is older.")
    print(f"Uses tensor cores: {torch.cuda.is_available()}")
    print()
    
    # Load model (U-Net part)
    if not use_custom_unet:
        # The load_trainable_model now returns the unet directly
        unet_model = load_trainable_model(network_pkl, device, dtype=dtype)
    else:
        unet_model = unet_model_custom()
    
    inception_model_instance = None
    if fid_eval_epochs > 0 and fid_ref_path:
        print("FID calculation is enabled. Attempting to load Inception model.")
        inception_model_instance, cifar10_ref = load_inception_model(device)
        if inception_model_instance is None:
            print("Failed to load Inception model. FID will be disabled.")
            # Effectively disable FID by not passing the model or path
            # The training function will also check for this.
    else:
        print("FID calculation is disabled (fid_eval_epochs=0 or "
              "fid_ref_path not provided).")

    # Dataset
    img_shape = (3, 32, 32) # Default for CIFAR/STL
    if dataset_name.lower() == "cifar10":
        print("Loading CIFAR-10 dataset...")
        train_loader, test_loader = load_cifar10_torch(
            batch_size, num_workers
        )
    elif dataset_name.lower() == "stl10":
        print("Loading and aligning STL10 dataset...")
        train_loader = create_aligned_stl10_dataloader(
            batch_size=batch_size, num_workers=num_workers
        )
        test_loader = create_aligned_stl10_test_dataloader(
            batch_size=batch_size, num_workers=num_workers
        )
        # STL10 images are 96x96, but create_aligned_stl10_dataloader resizes to 32x32
        # So img_shape remains (3, 32, 32)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Compile model if PyTorch 2.0+
    # if hasattr(torch, 'compile') and callable(torch.compile):
    #     print("Compiling model with torch.compile()...")
    #     try:
    #         unet_model = torch.compile(unet_model, mode="default") # Or "reduce-overhead"
    #         print("Model compiled successfully.")
    #     except Exception as e:
    #         print(f"torch.compile failed: {e}. Proceeding without compilation.")
    # else:
    #     print("torch.compile not available. Proceeding without compilation.")

    print(f"Loaded model and dataset for finetuning on '{dataset_name}'.")

    # Prepare W&B run name and config
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    current_run_name = f"{wandb_run_name_prefix}-{dataset_name}-{timestamp}"

    config_for_wandb = {
        "network_pkl": network_pkl,
        "dataset": dataset_name,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "eval_interval": eval_interval,
        "patience": patience,
        "num_workers": num_workers,
        "weight_decay": wd,
        "dtype": str(dtype),
        "device": str(device),
        "seed": seed,
        "fid_eval_epochs": fid_eval_epochs,
        "fid_num_samples": fid_num_samples,
        "fid_gen_batch_size": fid_gen_batch_size,
        "fid_inception_batch_size": fid_inception_batch_size,
        "fid_ref_path": fid_ref_path,
        "fid_num_workers": fid_num_workers,
        "use_custom_unet": use_custom_unet
    }

    # Finetune
    _ = train_flow_matching_edm_with_songunet(
        model=unet_model, # Pass the U-Net part
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=epochs,
        patience=patience,
        eval_interval=eval_interval,
        lr=learning_rate,
        wd=wd,
        device=str(device), # Pass device as string
        seed=seed,
        img_shape=img_shape,
        edm_params=None, # Use defaults or allow customization
        num_gen_samples_epoch_end=num_gen_samples_epoch_end, # Generate 10 samples
        num_inference_steps_epoch_end=num_inference_steps_epoch_end,
        dtype=dtype,
        # W&B specific args
        wandb_project=wandb_project_name,
        wandb_run_name=current_run_name,
        wandb_entity=wandb_entity,
        log_config=config_for_wandb,
        # Pass FID related parameters and the loaded Inception model
        inception_model=inception_model_instance,
        fid_ref=cifar10_ref,
        fid_eval_epochs=fid_eval_epochs,
        fid_num_samples=fid_num_samples,
        fid_gen_batch_size=fid_gen_batch_size,
        fid_inception_batch_size=fid_inception_batch_size,
        fid_ref_path=fid_ref_path,
        fid_num_workers=fid_num_workers,
        use_custom_unet=use_custom_unet,
        hf_repo=hf_repo,
    )
    print("Finetuning process finished.")

if __name__ == '__main__':
    # Example usage: Fine-tune on STL10
    # Make sure to log in to W&B first using `wandb login` in your terminal
    
    # Common pre-trained model for CIFAR-10 like data (32x32 conditional)
    cifar10_edm_pkl = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl'
    cifar10_fid_ref_path = "fid_refs/cifar10-32x32.npz"
    
    # To run CIFAR-10 finetuning (mainly for testing setup)
    train(
        network_pkl=cifar10_edm_pkl,
        dataset_name="cifar10",
        batch_size=128, # Adjust based on VRAM 160?
        learning_rate=8e-5,
        epochs=50, # For a quick test
        eval_interval=100000, # Evaluate more frequently for small datasets
        patience=10,
        num_workers=4,
        num_gen_samples_epoch_end = 64,
        num_inference_steps_epoch_end = 50,
        dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16,
        # dtype=torch.float32,
        wandb_project_name="my-edm-fm-tests", # Change to your project
        wandb_run_name_prefix="cifar10-fm-test",
        wandb_entity="YOUR-WANDB", # Set your entity
        fid_eval_epochs=2, # Evaluate FID every epoch for this test
        fid_num_samples=1000, # Small number for quick test
        fid_gen_batch_size=64,
        fid_inception_batch_size=128,
        fid_ref_path=cifar10_fid_ref_path, # Path to CIFAR10 FID stats
        fid_num_workers=2, # Simpler for small test
        use_custom_unet=False,
        hf_repo= "edm-fm",
    )

    # To run STL10 finetuning (your main goal for distribution shift)
    # train(
    #     network_pkl=cifar10_edm_pkl, # Using CIFAR-10 pre-trained model
    #     dataset_name="stl10",
    #     batch_size=128, # Adjust based on VRAM
    #     learning_rate=3e-5, # Potentially smaller LR for fine-tuning
    #     epochs=50, # Or more, depending on convergence
    #     eval_interval=100, # Eval interval relative to dataset size
    #     patience=10, # More patience for longer runs
    #     dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16, # Use bfloat16 if supported, else float16
    #     wandb_project_name="edm_fm_stl10_shift", # Your project
    #     wandb_run_name_prefix="stl10-fm-shift",
    #     # wandb_entity="your_wandb_username_or_team" # Set your entity
    #     seed=42 # Example seed
    # )


"""
Update the generate samples such that it generates a unique image with several generations like the example.py
Using bfloat16, tensor cores and w/o compile, with bs of 128, 1.33it/s, 1 epoch 5 min.
huggingface-cli login
wandb login
"""