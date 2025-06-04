### logging_utils.py ###
import wandb

# Store the run object globally within this module once initialized
_wandb_run = None

def init_wandb(project_name: str, run_name: str = None, config: dict = None, entity: str = None):
    """
    Initializes a new W&B run.

    Args:
        project_name (str): Name of the W&B project.
        run_name (str, optional): Name of the run. Defaults to None (W&B auto-generates).
        config (dict, optional): Hyperparameters and other run configurations.
        entity (str, optional): W&B entity (username or team name).
    """
    global _wandb_run
    try:
        if wandb.run is not None:
            print("W&B run already initialized. Finishing the current one.")
            wandb.finish()

        _wandb_run = wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            entity=entity,
            reinit=True # Allow reinitialization if called multiple times
        )
        print(f"W&B run initialized: {_wandb_run.url}")
    except Exception as e:
        print(f"Error initializing W&B: {e}. W&B logging will be disabled.")
        _wandb_run = None # Ensure it's None if init fails

def log_metrics(metrics_dict: dict, step: int = None, commit: bool = False):
    """
    Logs a dictionary of metrics to W&B with improved error handling.

    Args:
        metrics_dict (dict): Dictionary of metric_name: value.
        step (int, optional): The current step (e.g., batch or epoch).
                             If None, W&B uses its internal step.
        commit (bool): Whether to commit the log. Set to False if logging
                       multiple metrics for the same step incrementally.
    """
    if _wandb_run:
        try:
            # Ensure step is provided for custom metrics to avoid conflicts
            if step is not None:
                _wandb_run.log(metrics_dict, step=step, commit=commit)
            else:
                # Let wandb handle the step automatically for default metrics
                _wandb_run.log(metrics_dict, commit=commit)
        except Exception as e:
            print(f"Error logging metrics to W&B: {e}")
            print(f"Metrics: {metrics_dict}, Step: {step}")

def log_image(
    image_key: str,
    image_data,  # Can be PIL Image, numpy array, torch tensor, or path to image
    caption: str = None,
    step: int = None,
    commit: bool = False
):
    """
    Logs an image to W&B with explicit step and commit control.

    Args:
        image_key (str): Key for the image in W&B dashboard.
        image_data: The image data.
        caption (str, optional): Caption for the image.
        step (int, optional): The current step.
        commit (bool): Whether to commit the log immediately.
    """
    if _wandb_run:
        try:
            wandb_image = wandb.Image(image_data, caption=caption)
            if step is not None:
                _wandb_run.log({image_key: wandb_image}, step=step, 
                              commit=commit)
            else:
                _wandb_run.log({image_key: wandb_image}, commit=commit)
        except Exception as e:
            print(f"Error logging image to W&B: {e}")
            print(f"Image key: {image_key}, Step: {step}")

def finish_wandb():
    """Finishes the current W&B run."""
    global _wandb_run
    if _wandb_run:
        try:
            wandb.finish()
            print("W&B run finished.")
        except Exception as e:
            print(f"Error finishing W&B run: {e}")
        finally:
            _wandb_run = None # Reset after finishing

def is_wandb_initialized():
    """Checks if W&B has been successfully initialized."""
    return _wandb_run is not None
