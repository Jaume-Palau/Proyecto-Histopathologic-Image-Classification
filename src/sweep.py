
import wandb

from train import train_the_model
from src.config import sweep_config



################################################################################
## Train the model with sweeps W&B
################################################################################

def train_the_model_with_sweeps():

    with wandb.init() as run:
        config = wandb.config

        output = train_the_model(
            config=config,
            n_epochs=1,
            verbose=False,
            progress_print=False
        )



if __name__=="__main__":

        wandb.login()

        sweep_id = wandb.sweep(sweep_config, project="histopathologic-project") 
        
        wandb.agent(sweep_id, train_the_model_with_sweeps, count=1)