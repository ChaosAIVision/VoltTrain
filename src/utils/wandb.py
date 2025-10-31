import wandb

def create_wandb_run(project_name: str, run_name: str, config: dict = None):
    """
    Create a new WandB run

    Args:
        project_name (str): Name of the WandB project
        run_name (str): Name of the WandB run
        config (dict, optional): Configuration dictionary to log
    """
    wandb.init(
        project=project_name,
        name=run_name,
        config=config
    )