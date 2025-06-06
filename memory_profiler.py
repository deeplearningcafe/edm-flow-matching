import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import wandb # Assuming wandb is used as per original code
try:
    from training.fm import sample_lognorm_timesteps
except ImportError:
    raise("Import error")
try:
    from finetune import load_trainable_model, load_cifar10_torch, unet_model_custom
except ImportError:
    raise("Import error")
from datetime import datetime

# PyTorch Profiler imports
from torch.profiler import profile, record_function, ProfilerActivity, schedule

# Assuming these utils are available from your project
# from . import logging_utils # Placeholder for your logging_utils
# from .model_utils import create_songunet_lr_groups # Placeholder
# from .sampling_utils import sample_lognorm_timesteps # Placeholder

# Dummy logging_utils for standalone execution
class DummyLoggingUtils:
    def __init__(self):
        self._wandb_run = None
        self.wandb_initialized = False

    def init_wandb(self, project_name, run_name, config, entity):
        print(f"[Dummy] W&B init: project={project_name}, run={run_name}")
        # You might want to actually init wandb here if testing
        # try:
        #     self._wandb_run = wandb.init(project=project_name, name=run_name, config=config, entity=entity)
        #     self.wandb_initialized = True
        # except Exception as e:
        #     print(f"W&B initialization failed: {e}")
        #     self.wandb_initialized = False
        pass

    def is_wandb_initialized(self):
        return self.wandb_initialized # return self._wandb_run is not None

    def log_metrics(self, payload, step=None, commit=True):
        # if self.is_wandb_initialized():
        #     wandb.log(payload, step=step, commit=commit)
        # else:
        print(f"[Dummy] Log metrics (step {step}, commit {commit}): {payload}")
        pass

logging_utils = DummyLoggingUtils()

# Dummy model_utils
def create_songunet_lr_groups(model, base_lr, img_resolution, output_lr_multiplier, time_embed_lr_multiplier, backbone_lr_multiplier):
    print("[Dummy] create_songunet_lr_groups called")
    return [{'params': model.parameters(), 'lr': base_lr, 'name': 'all_params'}]


# Modified flow_matching_step with record_function for profiling
def flow_matching_step_profiled(
    unet_model: torch.nn.Module,
    x1: torch.Tensor,
    cond_input: torch.Tensor,
    generator: torch.Generator = None,
    input_perturbation: bool = False,
    edm_params: dict = None,
    unet_label_dim: int = 0,
    dtype=torch.float32,
):
    if edm_params is None:
        edm_params = {
            'sigma_min': 0.002, 'sigma_max': 80.0, 'sigma_data': 0.5
        }

    batch_size = x1.shape[0]
    device = x1.device

    with record_function("fm_step_sample_x0"):
        x0 = torch.randn_like(x1, dtype=x1.dtype)

    with record_function("fm_step_sample_t"):
        t = sample_lognorm_timesteps(batch_size, 0, 1, device, generator) # Expects (B,)

    with record_function("fm_step_interpolate_xt"):
        # Ensure t has shape (B, 1, 1, 1) for broadcasting
        t_reshaped = t.view(-1, 1, 1, 1)
        xt = (1 - t_reshaped) * x0 + t_reshaped * x1

    if input_perturbation:
        with record_function("fm_step_input_perturbation"):
            noise_scale = (1 - t_reshaped) * 0.1 # Use t_reshaped
            xt = xt + noise_scale * torch.randn_like(xt)

    with record_function("fm_step_calculate_v_true"):
        v_true = x1 - x0

    with record_function("fm_step_calculate_edm_sigma_conditioning"):
        sigma = edm_params['sigma_max'] * \
                (edm_params['sigma_min'] / edm_params['sigma_max']) ** t
        sigma_b = sigma.view(-1, 1, 1, 1)
        c_in_val = 1 / (edm_params['sigma_data']**2 + sigma_b**2).sqrt()
        unet_noise_labels = (sigma.log() / 4).flatten() # Should be (B,)

    unet_class_labels = None
    if unet_label_dim > 0:
        with record_function("fm_step_prepare_class_labels"):
            if cond_input is None:
                raise ValueError("cond_input is None but unet_label_dim > 0.")
            unet_class_labels = F.one_hot(
                cond_input.long(), num_classes=unet_label_dim
            ).to(dtype)

    with record_function("fm_step_unet_forward_pass"):
        v_pred = unet_model(
            x=xt * c_in_val,
            noise_labels=unet_noise_labels,
            class_labels=unet_class_labels,
            augment_labels=None
        )

    with record_function("fm_step_calculate_mse_loss"):
        loss = F.mse_loss(v_pred, v_true, reduction="mean")
    return loss

def flow_matching_step_min_rf(
    unet_model: torch.nn.Module,
    x1: torch.Tensor,
    cond_input: torch.Tensor,
    edm_params: dict = None,
    unet_label_dim: int = 0,
    dtype=torch.float32,
    log=False,
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
            with record_function("fm_step_prepare_class_labels"):
                if cond_input is None:
                    raise ValueError("cond_input is None but unet_label_dim > 0.")
                unet_class_labels = F.one_hot(
                    cond_input.long(), num_classes=unet_label_dim
                ).to(dtype)

        if log:torch.cuda.nvtx.range_push("forward")
        v_pred = unet_model(
            x=xt * c_in_val,
            noise_labels=unet_noise_labels,
            class_labels=unet_class_labels,
            augment_labels=None
        )
    else:
        if log:torch.cuda.nvtx.range_push("forward")
        v_pred = unet_model(
        xt * c_in_val,
        unet_noise_labels,
        encoder_hidden_states=cond_input.long().unsqueeze(-1).unsqueeze(-1),
        )
    if log:torch.cuda.nvtx.range_pop()

    batchwise_mse = ((x1 - x0 - v_pred) ** 2).mean(dim=list(range(1, len(x1.shape))))
    return batchwise_mse.mean()

def flow_matching_step_min_rf_profiled(
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
    
    with record_function("fm_step_sample_t"):
        nt = torch.randn((batch_size,)).to(x1.device)
        t = torch.sigmoid(nt)
        texp = t.view([batch_size, *([1] * len(x1.shape[1:]))])
    
    with record_function("fm_step_sample_x0"):
        x0 = torch.randn_like(x1)
    
    with record_function("fm_step_interpolate_xt"):
        xt = (1 - texp) * x0 + texp * x1

    with record_function("fm_step_calculate_edm_sigma_conditioning"):
        sigma = edm_params['sigma_max'] * \
                (edm_params['sigma_min'] / edm_params['sigma_max']) ** t
        sigma_b = sigma.view([batch_size, *([1] * len(x1.shape[1:]))])
        c_in_val = 1 / (edm_params['sigma_data']**2 + sigma_b**2).sqrt()
        unet_noise_labels = (sigma.log() / 4).flatten() # Should be (B,)

    if not use_custom_unet:
        unet_class_labels = None
        if unet_label_dim > 0:
            with record_function("fm_step_prepare_class_labels"):
                if cond_input is None:
                    raise ValueError("cond_input is None but unet_label_dim > 0.")
                unet_class_labels = F.one_hot(
                    cond_input.long(), num_classes=unet_label_dim
                ).to(dtype)

        with record_function("fm_step_unet_forward_pass"):
            v_pred = unet_model(
                x=xt * c_in_val,
                noise_labels=unet_noise_labels,
                class_labels=unet_class_labels,
                augment_labels=None
            )
    else:
        with record_function("fm_step_unet_forward_pass"):
            v_pred = unet_model(
                xt * c_in_val,
                unet_noise_labels,
                encoder_hidden_states=cond_input.long().unsqueeze(-1).unsqueeze(-1),
            )

    with record_function("fm_step_calculate_mse_loss"):
        batchwise_mse = ((x1 - x0 - v_pred) ** 2).mean(dim=list(range(1, len(x1.shape))))
    return batchwise_mse.mean()


# --- Profiler Trace Handler ---
def profiler_trace_handler(prof_instance, output_dir="profiler_output"):
    """
    Handles the trace when it's ready.
    Prints key averages and exports a Chrome trace file.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Print key averages, sorted by self CUDA time total
    # Note: The exact column names might vary slightly with PyTorch versions.
    # Adjust parsing if logging to wandb.Table more robustly.
    key_avg_table_str = prof_instance.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)
    print(f"\n--- PyTorch Profiler: Key Averages (Step {prof_instance.step_num}) ---")
    print(key_avg_table_str)
    key_avg_table_str = prof_instance.key_averages().table(sort_by="cpu_time_total", row_limit=20)
    print(f"\n--- PyTorch Profiler: Key Averages (Step {prof_instance.step_num}) ---")
    print(key_avg_table_str)

    # Export Chrome trace file
    timestamp = datetime.now().strftime(r'%Y%m%d-%H%M%S')
    trace_file_name = f"trace_step_{prof_instance.step_num}_{timestamp}.json"
    trace_file_path = os.path.join(output_dir, trace_file_name)
    prof_instance.export_chrome_trace(trace_file_path)
    print(f"Chrome trace saved to: {trace_file_path}")

    if logging_utils.is_wandb_initialized():
        # Log the table as text/HTML or the trace file as an artifact
        wandb.log({
            f"profiler/key_averages_step_{prof_instance.step_num}": wandb.Html(f"<pre>{key_avg_table_str}</pre>")
        }, step=prof_instance.step_num) # Use profiler's step num for log step
        wandb.save(trace_file_path, base_path=output_dir) # Save trace to W&B artifacts
        print(f"Trace file {trace_file_path} scheduled for W&B upload.")


def train_flow_matching_edm_with_songunet_profiler(
    model: torch.nn.Module,
    train_loader,
    epochs: int = 10,
    lr: float = 1e-4,
    wd: float = 1e-6,
    device: str = "cuda",
    seed: int = None,
    img_shape: tuple = (3, 512, 512),
    edm_params: dict = None,
    dtype=torch.bfloat16,
    log_config: dict = None,
    # --- Profiler specific arguments ---
    profile_gpu: bool = False,             # Enable GPU profiling
    profile_skip_steps: int = 5,       # App-level warmup steps before profiler starts recording cycle
    profile_warmup_steps: int = 1,     # Profiler's internal warmup phase (after skip, before active)
    profile_active_steps: int = 3,     # Number of steps to actively profile
    profile_repeat_cycles: int = 1,    # Number of times to repeat the (wait, warmup, active) cycle
    profile_output_dir: str = "profiler_output", # Directory to save traces
    use_custom_unet=False,
):
    if edm_params is None:
        edm_params = {
            'sigma_min': 0.002, 'sigma_max': 80.0, 'sigma_data': 0.5
        }

    if log_config is None:
        log_config = {
            "epochs": epochs, "learning_rate": lr, "weight_decay": wd,
            "batch_size": train_loader.batch_size if hasattr(train_loader, 'batch_size') else 'N/A',
            # ... (other params)
            "profile_gpu": profile_gpu,
            "profile_skip_steps": profile_skip_steps,
            "profile_warmup_steps": profile_warmup_steps,
            "profile_active_steps": profile_active_steps,
            "profile_repeat_cycles": profile_repeat_cycles,
        }

    model_label_dim = 0
    if hasattr(model, 'map_label') and model.map_label is not None:
        model_label_dim = model.map_label.in_features
    elif hasattr(model, 'label_dim'):
        model_label_dim = model.label_dim
    print(f"Using model_label_dim: {model_label_dim}")

    torch.cuda.empty_cache()
    img_resolution = img_shape[1]

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr, # LR is set in groups if they exist
        weight_decay=wd, fused=False
    )
    # ... (Scheduler setup - keep as is) ...
    if hasattr(train_loader, '__len__') and len(train_loader) > 0:
        t_max_steps = epochs * len(train_loader)
    else:
        print("Warning: train_loader length unknown, using estimated T_max.")
        estimated_steps_per_epoch = 500000 // (train_loader.batch_size if hasattr(train_loader, 'batch_size') and train_loader.batch_size else 16)
        t_max_steps = epochs * estimated_steps_per_epoch
        if t_max_steps == 0: t_max_steps = 10000
    
    warmup_steps_sched = round(0.02 * t_max_steps) if t_max_steps > 0 else 0
    warmup_steps_sched = min(warmup_steps_sched, t_max_steps -1) if t_max_steps > 0 else 0
    
    scheduler = None
    if t_max_steps > 0 and warmup_steps_sched < t_max_steps :
        scheduler_1 = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_steps_sched if warmup_steps_sched > 0 else 1
        )
        cosine_t_max = t_max_steps - warmup_steps_sched
        if cosine_t_max <= 0: cosine_t_max = 1
        scheduler_2 = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_t_max, eta_min=lr * 0.01
        )
        if warmup_steps_sched > 0:
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer, [scheduler_1, scheduler_2], milestones=[warmup_steps_sched]
            )
        else:
            scheduler = scheduler_2


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

    # --- Configure Profiler ---
    profiler_schedule_config = None
    on_trace_ready_handler = None

    if profile_gpu:
        os.makedirs(profile_output_dir, exist_ok=True)
        profiler_schedule_config = torch.profiler.schedule(
            skip_first=profile_skip_steps,    # Skip these many initial steps (application warmup)
            wait=0,                           # No idle steps between cycles for this simple setup
            warmup=profile_warmup_steps,      # Profiler's own warmup (after skip_first)
            active=profile_active_steps,      # Actively record these many steps
            repeat=profile_repeat_cycles      # Number of profiling cycles
        )
        # Use a lambda to pass the output directory to the handler
        on_trace_ready_handler = lambda prof_instance: profiler_trace_handler(prof_instance, profile_output_dir)
        print(f"PyTorch Profiler configured: App Skip Steps={profile_skip_steps}, "
              f"Profiler Warmup={profile_warmup_steps}, Active Record={profile_active_steps}, "
              f"Repeat Cycles={profile_repeat_cycles}")
        print(f"Profiler traces will be saved to: {profile_output_dir}")

    try:
        print(f"Starting Flow Matching training for SongUNet (EDM pre-trained)...")
        # Wrap the entire training operation with the profiler context
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if profile_gpu else [ProfilerActivity.CPU],
            schedule=profiler_schedule_config,    # schedule_config will be None if profile_gpu is False
            on_trace_ready=on_trace_ready_handler,# handler will be None if profile_gpu is False
            record_shapes=profile_gpu,            # Only record shapes if GPU profiling
            profile_memory=profile_gpu,           # Only profile memory if GPU profiling
            with_stack=profile_gpu                # Only include stack if GPU profiling
        ) as prof: # prof will be an actual profiler object if configured, else a dummy/None
            for epoch in range(epochs):
                model.train()
                epoch_loss_accum = 0.0
                num_batches_epoch = 0
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

                for batch_idx, batch in enumerate(progress_bar):
                    optimizer.zero_grad()

                    x1_batch = batch[0].to(dtype=dtype).to(device)
                    cond_input_batch = batch[1].to(dtype=dtype).to(device) if len(batch) > 1 else None
                    
                    # Use the flow_matching_step_profiled version
                    # loss = flow_matching_step_profiled(
                    #     unet_model=model,
                    #     x1=x1_batch,
                    #     cond_input=cond_input_batch,
                    #     generator=generator,
                    #     input_perturbation=True,
                    #     edm_params=edm_params,
                    #     unet_label_dim=model_label_dim,
                    #     dtype=dtype
                    # )
                    loss = flow_matching_step_min_rf_profiled(
                        unet_model=model,
                        x1=x1_batch,
                        cond_input=cond_input_batch,
                        edm_params=edm_params,
                        unet_label_dim=model_label_dim,
                        dtype=dtype,
                        use_custom_unet=use_custom_unet
                    )

                    with record_function("fm_check_nan"):
                        # it calls cudaStreamSynchronize but there is not performance diff also this torch ops should be on gpu
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"Warning: NaN/Inf loss detected at step {global_step}. Skipping.")
                            # If profiler is active, still call step to maintain schedule
                            if profile_gpu and prof: prof.step()
                            global_step += 1
                            continue
                    with record_function("fm_backward"):
                        loss.backward()
                    with record_function("fm_clip_grad"): 
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    if scheduler:
                        scheduler.step()

                    epoch_loss_accum += loss.item()
                    num_batches_epoch += 1

                    # Signal the profiler that a step is complete
                    if profile_gpu and prof:
                        prof.step()
                    
                    global_step += 1
                    
                    # Optional: Stop training after the first profiling cycle for quick analysis
                    if profile_gpu and profile_repeat_cycles == 1 and \
                       prof and prof.step_num >= profile_skip_steps + profile_warmup_steps + profile_active_steps:
                        print("First profiling cycle complete. Exiting for analysis as profile_repeat_cycles is 1.")
                        return # Exit the training function

                # End of epoch
                avg_epoch_loss = epoch_loss_accum / num_batches_epoch if num_batches_epoch > 0 else 0
                print(f"End of Epoch {epoch+1}: Avg Train Loss: {avg_epoch_loss:.4f}")

    except StopIteration:
        print("Training stopped early due to patience.")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        print("Training finished.")

def train_flow_matching_edm_with_songunet_min_rf(
    model: torch.nn.Module,
    train_loader,
    epochs: int = 10,
    lr: float = 1e-4,
    wd: float = 1e-6,
    device: str = "cuda",
    seed: int = None,
    img_shape: tuple = (3, 512, 512),
    edm_params: dict = None,
    dtype=torch.bfloat16,
    warmup_iters=10,
    use_custom_unet=False,
):
    if edm_params is None:
        edm_params = {
            'sigma_min': 0.002, 'sigma_max': 80.0, 'sigma_data': 0.5
        }

    model_label_dim = 0
    if hasattr(model, 'map_label') and model.map_label is not None:
        model_label_dim = model.map_label.in_features
    elif hasattr(model, 'label_dim'):
        model_label_dim = model.label_dim
    print(f"Using model_label_dim: {model_label_dim}")

    torch.cuda.empty_cache()
    img_resolution = img_shape[1]

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr, # LR is set in groups if they exist
        weight_decay=wd, fused=False
    )
    # ... (Scheduler setup - keep as is) ...
    if hasattr(train_loader, '__len__') and len(train_loader) > 0:
        t_max_steps = epochs * len(train_loader)
    else:
        print("Warning: train_loader length unknown, using estimated T_max.")
        estimated_steps_per_epoch = 500000 // (train_loader.batch_size if hasattr(train_loader, 'batch_size') and train_loader.batch_size else 16)
        t_max_steps = epochs * estimated_steps_per_epoch
        if t_max_steps == 0: t_max_steps = 10000
    
    warmup_steps_sched = round(0.02 * t_max_steps) if t_max_steps > 0 else 0
    warmup_steps_sched = min(warmup_steps_sched, t_max_steps -1) if t_max_steps > 0 else 0
    
    scheduler = None
    if t_max_steps > 0 and warmup_steps_sched < t_max_steps :
        scheduler_1 = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=warmup_steps_sched if warmup_steps_sched > 0 else 1
        )
        cosine_t_max = t_max_steps - warmup_steps_sched
        if cosine_t_max <= 0: cosine_t_max = 1
        scheduler_2 = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_t_max, eta_min=lr * 0.01
        )
        if warmup_steps_sched > 0:
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer, [scheduler_1, scheduler_2], milestones=[warmup_steps_sched]
            )
        else:
            scheduler = scheduler_2


    generator = torch.Generator(device=device)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        generator.manual_seed(seed)
    
    model.to(device)
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

                # start profiling after 10 warmup iterations
                if global_step == warmup_iters: torch.cuda.cudart().cudaProfilerStart()
                # push range for current iteration
                if global_step >= warmup_iters: torch.cuda.nvtx.range_push("iteration{}".format(global_step))

                x1_batch = batch[0].to(dtype=dtype).to(device)
                cond_input_batch = batch[1].to(dtype=dtype).to(device) if len(batch) > 1 else None
                # Use the flow_matching_step_profiled version
                loss = flow_matching_step_min_rf(
                    unet_model=model,
                    x1=x1_batch,
                    cond_input=cond_input_batch,
                    edm_params=edm_params,
                    unet_label_dim=model_label_dim,
                    dtype=dtype,
                    log=global_step >= warmup_iters,
                    use_custom_unet=use_custom_unet,
                )

                # it calls cudaStreamSynchronize but there is not performance diff also this torch ops should be on gpu
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN/Inf loss detected at step {global_step}. Skipping.")

                if global_step >= warmup_iters: torch.cuda.nvtx.range_push("backward")
                loss.backward()
                if global_step >= warmup_iters: torch.cuda.nvtx.range_pop()

                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if global_step >= warmup_iters: torch.cuda.nvtx.range_push("opt.step()")
                optimizer.step()
                if global_step >= warmup_iters: torch.cuda.nvtx.range_pop()
                if scheduler:
                    scheduler.step()

                epoch_loss_accum += loss.item()
                num_batches_epoch += 1

                
                global_step += 1
                if global_step >= warmup_iters: torch.cuda.nvtx.range_pop()

                # Optional: Stop training after the first profiling cycle for quick analysis
                if global_step > warmup_iters+5:
                    print("First profiling cycle complete. Exiting for analysis as profile_repeat_cycles is 1.")
                    return # Exit the training function

            # End of epoch
            avg_epoch_loss = epoch_loss_accum / num_batches_epoch if num_batches_epoch > 0 else 0
            print(f"End of Epoch {epoch+1}: Avg Train Loss: {avg_epoch_loss:.4f}")

    except StopIteration:
        print("Training stopped early due to patience.")
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        torch.cuda.cudart().cudaProfilerStop()
        print("Training finished.")


if __name__ == "__main__":
    seed = 46
    torch.manual_seed(seed)
    np.random.seed(seed) # Ensure all seeds are set

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using device: {device}")

    # Load model (U-Net part)
    # The load_trainable_model now returns the unet directly
    network_pkl = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl'
    dtype = torch.float32
    # unet_model = load_trainable_model(network_pkl, device, dtype=dtype)
    unet_model = unet_model_custom()
    batch_size = 16
    num_workers=4
    # Dataset
    img_shape = (3, 32, 32) # Default for CIFAR/STL
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = load_cifar10_torch(
        batch_size, num_workers
    )
    use_profiler = False
    use_custom_unet = False
    if use_profiler:
        train_flow_matching_edm_with_songunet_profiler(
        model=unet_model, # Pass the U-Net part
        train_loader=train_loader,
        epochs=2,
        lr=5e-5,
        wd=0.01,
        device=str(device), # Pass device as string
        seed=seed,
        img_shape=img_shape,
        edm_params=None, # Use defaults or allow customization
        dtype=dtype,
        # --- Profiler specific arguments ---
        profile_gpu=True,
        profile_skip_steps=6,    # e.g., skip first 5 global steps
        profile_warmup_steps=3,  # 1 profiler warmup step
        profile_active_steps=3,  # Record 3 steps
        profile_repeat_cycles=1, # Just one profiling session
        profile_output_dir="my_profiler_traces",
        use_custom_unet=use_custom_unet,
        )
        print("End")
    else:
        train_flow_matching_edm_with_songunet_min_rf(
        model=unet_model, # Pass the U-Net part
        train_loader=train_loader,
        epochs=2,
        lr=5e-5,
        wd=0.01,
        device=str(device), # Pass device as string
        seed=seed,
        img_shape=img_shape,
        edm_params=None, # Use defaults or allow customization
        dtype=dtype,
        warmup_iters=10,
        use_custom_unet=use_custom_unet,
        )

"""
sudo env "PATH=$PATH" nsys profile --gpu-metrics-devices=0 -t cuda,nvtx,osrt,cudnn,cublas --cudabacktrace=true --capture-range=cudaProfilerApi  --capture-range-end=stop python memory_profiler.py
"""