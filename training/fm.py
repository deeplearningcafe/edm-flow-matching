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
import torch.nn.functional as F
import numpy as np
try:
    from . import logging_utils # if logging_utils is in the same package
except ImportError:
    import logging_utils # if logging_utils is a top-level module
import wandb
try:
    from training.plot_utils import plot_generation_steps_with_grid, generate_samples_edm_fm
except ImportError:
    from plot_utils import plot_generation_steps_with_grid, generate_samples_edm_fm
try:
    from training.fid import evaluate_fid
except ImportError:
    from fid import evaluate_fid


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

def flow_matching_step_min_rf(
    unet_model: torch.nn.Module,
    x1: torch.Tensor,
    cond_input: torch.Tensor,
    edm_params: dict = None,
    unet_label_dim: int = 0,
    dtype=torch.float32,
    use_custom_unet=False
):
    if edm_params is None:
        edm_params = {
            'sigma_min': 0.002, 'sigma_max': 80.0, 'sigma_data': 0.5
        }

    batch_size = x1.shape[0]

    nt = torch.randn((batch_size,)).to(x1.device)
    t = torch.sigmoid(nt)

    texp = t.view([batch_size, *([1] * len(x1.shape[1:]))])
    x0 = torch.randn_like(x1)
    xt = (1 - texp) * x0 + texp * x1


    sigma = edm_params['sigma_max'] * \
            (edm_params['sigma_min'] / edm_params['sigma_max']) ** t
    sigma_b = sigma.view([batch_size, *([1] * len(x1.shape[1:]))])
    c_in_val = 1 / (edm_params['sigma_data']**2 + sigma_b**2).sqrt()
    unet_noise_labels = (sigma.log() / 4).flatten() # Should be (B,)
    
    if not use_custom_unet:
        unet_class_labels = None
        if unet_label_dim > 0:
                if cond_input is None:
                    raise ValueError("cond_input is None but unet_label_dim > 0.")
                unet_class_labels = F.one_hot(
                    cond_input.long(), num_classes=unet_label_dim
                ).to(dtype)

        v_pred = unet_model(
            x=xt * c_in_val,
            noise_labels=unet_noise_labels,
            class_labels=unet_class_labels,
            augment_labels=None
        )
    else:
        v_pred = unet_model(
            xt * c_in_val,
            unet_noise_labels,
            encoder_hidden_states=cond_input.long().unsqueeze(-1).unsqueeze(-1),
        )
    batchwise_mse = ((x1 - x0 - v_pred) ** 2).mean(dim=list(range(1, len(x1.shape))))
    return batchwise_mse.mean()

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
        new_name = name
        # support for compiled models
        if name.startswith("_orig_mod."):
            # Strip the prefix
            new_name = name[len("_orig_mod."):]

        is_output = new_name in output_param_names_exact
        is_time_embed = any(
            re.match(p, new_name) for p in time_embed_patterns_regex
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
    fid_ref=None,
    fid_eval_epochs: int = 0, # Default to 0 (disabled)
    fid_num_samples: int = 10000, 
    fid_gen_batch_size: int = 64, 
    fid_inception_batch_size: int = 64, 
    fid_ref_path: str = None, 
    fid_num_workers: int = 2,
    use_custom_unet=False,
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

    # Define custom metrics after wandb initialization
    if logging_utils.is_wandb_initialized():
        # Define custom x-axes for different metric types
        wandb.define_metric("epoch")
        wandb.define_metric("eval_step") 
                
        # Epoch-level metrics use epoch as x-axis
        wandb.define_metric("train/avg_epoch_loss", step_metric="epoch")
        wandb.define_metric("eval/fid_score", step_metric="epoch")
        wandb.define_metric("epoch_samples/*", step_metric="epoch")
        
        # Evaluation interval metrics use eval_step as x-axis
        wandb.define_metric("eval/avg_val_loss", step_metric="eval_step")
        wandb.define_metric("train/avg_loss_interval", step_metric="eval_step")
        wandb.define_metric("eval/epoch_progress_at_eval", 
                           step_metric="eval_step")


    # Infer label_dim from the SongUNet model
    # SongUNet stores label_dim directly or via self.map_label
    model_label_dim = 0
    if hasattr(model, 'map_label') and model.map_label is not None:
        model_label_dim = model.map_label.in_features
    elif hasattr(model, 'label_dim'): # If stored directly
        model_label_dim = model.label_dim
    print(f"Using model_label_dim: {model_label_dim}")
    # if logging_utils.is_wandb_initialized():
    #     logging_utils.log_metrics({"model/label_dim": model_label_dim}, commit=False)

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
    eval_step_counter = 0 
    
    try:
        print(f"Starting Flow Matching training for SongUNet (EDM pre-trained)...")
        for epoch in range(epochs):
            model.train()
            epoch_loss_accum = torch.tensor([0.0], device=device, dtype=dtype)
            num_batches_epoch = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

            for batch_idx, batch in enumerate(progress_bar):
                optimizer.zero_grad()

                x1_batch = batch[0].to(dtype=dtype).to(device)
                cond_input_batch = batch[1].to(dtype=dtype).to(device) if len(batch) > 1 else None

                # loss = flow_matching_step(
                #     unet_model=model,
                #     x1=x1_batch,
                #     cond_input=cond_input_batch,
                #     generator=generator,
                #     input_perturbation=True, # Often good for FM
                #     edm_params=edm_params,
                #     unet_label_dim=model_label_dim,
                #     dtype=dtype
                # )
                loss = flow_matching_step_min_rf(
                    unet_model=model,
                    x1=x1_batch,
                    cond_input=cond_input_batch,
                    edm_params=edm_params,
                    unet_label_dim=model_label_dim,
                    dtype=dtype,
                    use_custom_unet=use_custom_unet
                )

                # it calls cudaStreamSynchronize but there is not performance diff also this torch ops should be on gpu
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

                # returns the grad norm like the code above
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler:
                    scheduler.step()

                epoch_loss_accum += loss.detach()
                num_batches_epoch += 1

                if logging_utils.is_wandb_initialized():
                    log_payload = {
                        "train/loss_step": loss.item(),
                        "train/grad_norm_step": norm.item(),
                        "learning_rate": optimizer.param_groups[0]['lr']
                    }
                    # Log LR for each group if multiple groups exist
                    for i, pg in enumerate(optimizer.param_groups):
                        log_payload[f"lr_group_{pg.get('name', i)}"] = pg['lr']

                    logging_utils.log_metrics(log_payload, step=global_step, commit=False)

                # Evaluation and logging
                if (batch_idx + 1) % eval_interval == 0 or \
                   (batch_idx + 1) == len(train_loader):
                    # Increment custom step counter for evaluation logs
                    eval_step_counter += 1
                    avg_train_loss_interval = (epoch_loss_accum / num_batches_epoch).item()
                    
                    model.eval()
                    val_loss = torch.tensor([0.0], device=device, dtype=dtype)
                    with torch.no_grad():
                        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
                            x1_batch = batch[0].to(dtype=dtype).to(device)
                            cond_input_batch = batch[1].to(dtype=dtype).to(device) if len(batch) > 1 else None
                            loss = flow_matching_step_min_rf(
                                unet_model=model,
                                x1=x1_batch,
                                cond_input=cond_input_batch,
                                edm_params=edm_params,
                                unet_label_dim=model_label_dim,
                                dtype=dtype,
                                use_custom_unet=use_custom_unet
                            )
                            val_loss += loss.detach()

                    val_loss = (val_loss / len(test_loader)).item()

                    print(f"\nEpoch {epoch+1}, Batch {batch_idx+1}/"
                          f"{len(train_loader)}, EvalStep {eval_step_counter}: "
                          f"Train Loss Interval: {avg_train_loss_interval:.4f}, "
                          f"Val Loss: {val_loss:.4f}")

                    if logging_utils.is_wandb_initialized():
                        current_epoch_progress = epoch + \
                            (batch_idx + 1) / len(train_loader)
                        # Log eval metrics against custom 'eval_step'
                        logging_utils.log_metrics({
                            "eval/avg_val_loss": val_loss,
                            "train/avg_loss_interval": avg_train_loss_interval,
                            "eval/epoch_progress_at_eval": current_epoch_progress,
                            "eval_step": eval_step_counter
                        }, step=global_step, commit=False)

                    # Checkpointing and Early Stopping
                    if val_loss < best_eval_loss:
                        best_eval_loss = val_loss
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
                global_step += 1

            # --- End of Epoch ---
            avg_epoch_loss = epoch_loss_accum / num_batches_epoch if num_batches_epoch > 0 else float('inf')
            print(f"End of Epoch {epoch+1}/{epochs}: Avg Train Loss: {avg_epoch_loss:.4f}")
            if logging_utils.is_wandb_initialized():
                logging_utils.log_metrics({
                    "train/avg_epoch_loss": avg_epoch_loss,
                    "epoch": epoch,
                }, step=global_step, commit=False)

            # Generate and log samples at the end of each epoch
            if num_gen_samples_epoch_end > 0:
                print("Generating samples for W&B logging...")
                model.eval()
                # Generate for a few fixed labels if conditional, or unconditionally
                gen_labels = None
                if model_label_dim > 0:
                    # Ensure labels are within the valid range [0, model_label_dim - 1]
                    # num_labels_to_gen = min(model_label_dim, num_gen_samples_epoch_end)
                    if num_gen_samples_epoch_end <= model_label_dim:
                        # If batch size <= number of classes, use sequential labels
                        gen_labels = torch.arange(num_gen_samples_epoch_end, device=device)
                    else:
                        # If batch size > number of classes, repeat the sequence
                        base_labels = torch.arange(model_label_dim, device=device)
                        repeats = (num_gen_samples_epoch_end + model_label_dim - 1) // model_label_dim
                        gen_labels = base_labels.repeat(repeats)[:num_gen_samples_epoch_end]
                    
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

                    saved_plot_path = plot_generation_steps_with_grid(
                        traj, 
                        batch_labels=gen_labels,
                        num_classes=model_label_dim if model_label_dim > 0 else 10,
                        filename=full_plot_path,
                        title=f"FM Generation Results - Epoch {epoch+1}",
                        epoch=epoch+1,
                        eval_interval=eval_step_counter, 
                        gridw=8, 
                        gridh=8)

                    if saved_plot_path and logging_utils.is_wandb_initialized():
                        # Log trajectory and grid separately if they exist
                        traj_path = saved_plot_path.replace('.png', '_trajectory.png')
                        grid_path = saved_plot_path.replace('.png', '_grid.png')
                        
                        if os.path.exists(traj_path):
                            logging_utils.log_image(
                                "epoch_samples/generation_trajectory",
                                traj_path,
                                caption=f"Trajectory at Epoch {epoch+1}",
                                step=global_step,
                            )
                        
                        if os.path.exists(grid_path):
                            logging_utils.log_image(
                                "epoch_samples/grid",
                                grid_path,
                                caption=f"Generated Grid at Epoch {epoch+1}",
                                step=global_step,
                            )

        
            perform_fid = (
                fid_ref_path and 
                inception_model is not None and # Check if model was passed
                (epoch + 1) % fid_eval_epochs == 0 and 
                fid_num_samples > 0 and
                fid_eval_epochs > 0 # Ensure fid_eval_epochs is positive
            )

            if perform_fid:
                evaluate_fid(
                    model=model,
                    epoch=epoch,
                    global_step=global_step,
                    fid_ref=fid_ref,
                    inception_model=inception_model,
                    logging_utils=logging_utils,
                    fid_num_samples=fid_num_samples,
                    fid_gen_batch_size=fid_gen_batch_size,
                    fid_inception_batch_size=fid_inception_batch_size,
                    fid_num_workers=fid_num_workers,
                    # Model and generation parameters
                    model_label_dim=model_label_dim,
                    device=device,
                    seed=seed,
                    num_inference_steps_epoch_end=num_inference_steps_epoch_end,
                    img_shape=img_shape,
                    edm_params=edm_params,
                    dtype=dtype
                )
            elif (epoch + 1) % fid_eval_epochs == 0 and fid_eval_epochs > 0:
                # Log why FID was skipped if it was supposed to run
                reason = []
                if not fid_ref_path: reason.append("fid_ref_path not set")
                if inception_model is None: reason.append("Inception model not loaded")
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
        logging_utils.log_metrics({}, step=global_step, commit=True)


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
            # saved as an ordered dict
            torch.save(model.state_dict(), final_save_path)
            wandb.save(final_save_path, base_path=os.path.dirname(final_save_path))
            print(f"Final model saved to {final_save_path} and W&B.")
            logging_utils.finish_wandb()

    # Return model and any tracked local metrics if needed outside W&B
    return model # Placeholder for local metrics if you re-enable them
