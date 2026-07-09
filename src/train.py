
import os, pickle,time, torch
import numpy as np

from sklearn.metrics import roc_auc_score
import wandb
from config import *
from pathlib import Path
from tqdm.autonotebook import tqdm
from dataset import DatasetWrapper_Train, train_test_split
from helper_functions import plot_results
from model import CustomCNN
from model_more_spatial import CustomCNN_MoreSpatial


################################################################################
## Train the model
################################################################################

def train_the_model(
        model: CustomCNN|CustomCNN_MoreSpatial,
        config:dict,
        n_epochs:int,
        output_type:str,
        verbose:bool=False,
        progress_print:bool=False,
        use_sweeps=False,
    ):


    """Training loop that connects everything! See comments for details.

    Args: 
        model (CustomCNN | CustomCNN_MoreSpatial): The model to train.
        config (dict): Dictionary holding various parameters for the model
        n_epochs (int): Number of epochs to train and validate.
        verbose (bool): Whether to print out the profiling outputs.
        output_type (str): Type of model output.
            - "log_probs": model returns log-probabilities, e.g. LogSoftmax + NLLLoss.
            - "logits": model returns raw logits, e.g. CrossEntropyLoss.
    """

    ## Determine whether to use CPU/GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"|| Using {device} ||")
    
    ## Preparing dataset and train/test split it: 
    dataset_to_split = DatasetWrapper_Train()
    training_set, testing_set = train_test_split(dataset_to_split, train_proportion=0.8)
    

    model = model.to(device)  # Move model to the device
    ## Get the optimizer from model instance
    optimizer = model.get_optimizer(                   # Get the optimizer (ADAM)
        lr=config["lr"],                           # Get the learning rate from tunning config
        betas=(config["beta1"], config["beta2"]),  # Get the beta from tunning config 
        weight_decay=config["weight_decay"],       # Get the weight decay from tunning config
    )
    ## Get te loss func from the model instance
    loss_function = model.get_loss_function(reduction='sum')
    
    ## Create dataloader - for convenience
    training_dataloader = torch.utils.data.DataLoader(
        dataset=training_set, 
        batch_size=config["batch_size"], 
        shuffle=True,
        num_workers=config['dataloader_worker_count'],
        prefetch_factor=config["dataloader_prefetch_factor"],
        drop_last=True,  # The last few data that doesn't form a full batch gets dropped.
        pin_memory=True,  # Pin memory for faster transfer to GPU
    )
    validation_dataloader = torch.utils.data.DataLoader(
        dataset=testing_set, 
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config['dataloader_worker_count'],
        prefetch_factor=config["dataloader_prefetch_factor"],
        drop_last=True,  # The last few data that doesn't form a full batch gets dropped.
        pin_memory=True,  # Pin memory for faster transfer to GPU
    )
    
    training_dataset_size = len(training_dataloader.dataset)
    training_batches_per_epoch = int(training_dataset_size / config['batch_size'])  # Round down because drop_last==True
    
    validation_dataset_size = len(validation_dataloader.dataset)
    validation_batches_per_epoch = int(validation_dataset_size / config['batch_size'])  # Round down because drop_last==True

    ## Various tracker for data output
    track_training_loss     = np.zeros((n_epochs, training_batches_per_epoch), dtype=np.float64)
    track_training_TP_count = np.zeros((n_epochs, training_batches_per_epoch), dtype=np.int64)
    track_training_TN_count = np.zeros((n_epochs, training_batches_per_epoch), dtype=np.int64)
    track_training_FP_count = np.zeros((n_epochs, training_batches_per_epoch), dtype=np.int64)
    track_training_FN_count = np.zeros((n_epochs, training_batches_per_epoch), dtype=np.int64)
    
    track_validation_loss     = np.zeros((n_epochs, validation_batches_per_epoch), dtype=np.float64)
    track_validation_TP_count = np.zeros((n_epochs, validation_batches_per_epoch), dtype=np.int64)
    track_validation_TN_count = np.zeros((n_epochs, validation_batches_per_epoch), dtype=np.int64)
    track_validation_FP_count = np.zeros((n_epochs, validation_batches_per_epoch), dtype=np.int64)
    track_validation_FN_count = np.zeros((n_epochs, validation_batches_per_epoch), dtype=np.int64)
    
    
    best_val_auc_roc = -1
    patience = 5
    patience_counter = 0
    ## Main Training / Validation Loop
    for epoch in tqdm(range(n_epochs), desc="Epochs..."):  # Default to one epoch (because dataset is huge!!!)
        if progress_print: print(f"{epoch} / {n_epochs} ", "#"*50)
        ########## TRAINING PORTION ##########
        total_training_loop_start = time.time()  # Timer
        for batch_idx, (images, labels) in enumerate(tqdm(training_dataloader, desc="TRAINING PORTION")):
            if verbose: start_time = time.time()  # Timer

            if progress_print: print(f"{batch_idx} / {training_batches_per_epoch}", "#"*50)  # Timer
            if verbose: print(f"Timer - Load Dataloader batch (Train) : {time.time() - start_time}")  # Timer
            if verbose: start_time = time.time()  # Timer
            images = images.to(device)
            labels = labels.to(device)
            if verbose: print(f"Timer - Move to {device} : {time.time() - start_time}")  # Timer
            #print(f"Images Device: {images.device}, Labels Device: {labels.device}")  # Debug use

            ## Set model to training mode
            if verbose: start_time = time.time()  # Timer


            model.train()
            outputs = model(images)  # Inference: The model outputs are in log(proba) scale
            max_value, max_idx = torch.max(outputs, dim=1)
            prediction = max_idx  # Classification 
            ## Calculate metrics
            loss = loss_function(outputs, labels)  # Calculate the loss


            ## Backprop
            optimizer.zero_grad()  # Zero out the loss gradient - https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam.zero_grad
            loss.backward()        # Back propagate the loss
            optimizer.step()       # Perform a single optimization step - https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam
            if verbose: print(f"Timer - Train+Backprop : {time.time() - start_time}")  # Timer
            ## Update variables
            num_of_batches_trained = batch_idx + 1
            ## Update trackers
            if progress_print: print("Training loss: ", loss.item())  # Debug use
            track_training_loss[epoch, batch_idx] = loss.item()
            track_training_TP_count[epoch, batch_idx] = ((prediction==1) & (labels==1)).sum().item()
            track_training_TN_count[epoch, batch_idx] = ((prediction==0) & (labels==0)).sum().item()
            track_training_FP_count[epoch, batch_idx] = ((prediction==1) & (labels==0)).sum().item()
            track_training_FN_count[epoch, batch_idx] = ((prediction==0) & (labels==1)).sum().item()
        total_training_loop_time = time.time() - total_training_loop_start  # Timer
        print(f"Timer - ENTIRE TRAINING PORTION : {total_training_loop_time}")

        all_val_labels = []
        all_val_probs = []

        ## Validation Loop
        model.eval()
        print("Validation loop")
        total_validation_loop_start = time.time()  # Timer
        for batch_idx, (images, labels) in enumerate(tqdm(validation_dataloader), desc="VALIDATION PORTION"):

            if progress_print: 
                print(f"{batch_idx} / {validation_batches_per_epoch}", "#"*50)  # Timer

            with torch.no_grad():  # No gradient mode
                start_time = time.time()
                images = images.to(device)
                labels = labels.to(device)

                if verbose: 
                    print(f"Timer - Move to {device} : {time.time() - start_time}")
                    start_time =time.time()  # Timer

                ## Model inference
                outputs = model(images)

                max_value, max_idx = torch.max(outputs, dim=1)
                prediction = max_idx

                if output_type == "log_probs":
                    probs = torch.exp(outputs)
                    probs = probs[:,1]

                elif output_type == "logits":
                    probs = torch.softmax(outputs, dim=1)
                    probs = probs[:,1]


                all_val_labels.append(labels.detach().cpu())
                all_val_probs.append(probs.detach().cpu())

                if verbose: 
                    print(f"Timer - Model inference : {time.time() - start_time}")  # Timer

            ## Calculate metrics
            loss = loss_function(outputs, labels)
            ## Update variables
            num_of_batches_validated = batch_idx + 1
            ## Update trackers
            if progress_print: 
                print("Validation loss: ", loss.item())  # Debug use

            track_validation_loss[epoch, batch_idx] = loss.item()
            track_validation_TP_count[epoch, batch_idx] = ((prediction==1) & (labels==1)).sum().item()
            track_validation_TN_count[epoch, batch_idx] = ((prediction==0) & (labels==0)).sum().item()
            track_validation_FP_count[epoch, batch_idx] = ((prediction==1) & (labels==0)).sum().item()
            track_validation_FN_count[epoch, batch_idx] = ((prediction==0) & (labels==1)).sum().item()
        total_validation_loop_time = time.time() - total_validation_loop_start
        print(f"Timer - ENTIRE VALIDATION PORTION : {total_validation_loop_time}")

        y_true = torch.cat(all_val_labels).numpy()
        y_probs = torch.cat(all_val_probs).numpy()

        val_auc_roc = roc_auc_score(y_true,y_probs) # Calcula la metrica objetivo del concurso

        if use_sweeps:
            wandb.log({
            "AUC-ROC" : val_auc_roc,
            "Training Loss": track_training_loss, 
            "Validation Loss": track_validation_loss,
            })


        ########## CHECKPOINT PORTION ##########
        ## Create checkpoint data to be serialized
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_auc_roc": val_auc_roc,
            "config": dict(config),
        }

        #### GUARDAR MEJOR MODELO ####
        if val_auc_roc > best_val_auc_roc:
            best_val_auc_roc = val_auc_roc

            patience_counter = 0

            model_dir = Path(MODELS_DIR, config['name'])
            os.makedirs(model_dir, exist_ok=True)    # Make sure the directory is created
            model_path = Path(model_dir,"best_model.pt")
            torch.save(checkpoint_data, model_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

        ## Collect all the items into dictionary to return
        output = {
            "AUC-ROC" : best_val_auc_roc,
            "Training Loss": track_training_loss, 
            "Training TP": track_training_TP_count, 
            "Training FP": track_training_FP_count, 
            "Training TN": track_training_TN_count,
            "Training FN": track_training_FN_count,
            "Validation Loss": track_validation_loss, 
            "Validation TP": track_validation_TP_count, 
            "Validation FP": track_validation_FP_count, 
            "Validation TN": track_validation_TN_count,
            "Validation FN": track_validation_FN_count,
            #"loss": 1, # Dummy loss
        }

        ## Define the location to save(chekpoint)
        dir_path = Path(OUTPUTS_DIR, "checkpoints", config['name'])
        print(f"Checkpoint saved to {dir_path}.")
        os.makedirs(dir_path, exist_ok=True)    # Make sure the directory is created
        ## Serialize and save the checkpoint data
        checkpoint_path = Path(dir_path, "last_checkpoint.pt")
        torch.save(checkpoint_data, checkpoint_path)  # Serialize the checkpoint data
    
    dir_path = Path(OUTPUTS_DIR, "metrics",config['name'])
    os.makedirs(dir_path, exist_ok=True)    # Make sure the directory is created
    with open(Path(dir_path, "output.pkl"), "wb") as f: 
        pickle.dump(pickle.dumps(output), f)
   

    plot_results(config, output['Training Loss'], output['Validation Loss'], Path(os.getcwd(), config['name']))
    
    if True:  # Final report
        print(f"Timer - ENTIRE TRAINING PORTION : {total_training_loop_time}")
        print(f"Timer - ENTIRE VALIDATION PORTION : {total_validation_loop_time}")
        print(f"Checkpoint path: {checkpoint_path}")
        print(f"Model path: {model_path}")
        print(f"Pickle path: {dir_path, 'output.pkl'}")
        
        
    return output


if __name__ == "__main__":

    # output = train_the_model(
    #     config=best_model_config,
    #     n_epochs=20,
    #     verbose=False,
    #     progress_print=False,
    #     use_sweeps=False,
    #     model = CustomCNN_MoreSpatial(),
    #     output_type = "logits",
    # )

    output = train_the_model(
        config=best_model_config,
        n_epochs=20,
        verbose=False,
        progress_print=False,
        use_sweeps=False,
        model = CustomCNN(),
        output_type = "log_probs",
    )


    
    print(f'AUC-ROC-val : {output["AUC-ROC"]}',
          f'Validation Loss: {output["Validation Loss"].mean()}',
          f'Trainning Loss: {output["Training Loss"].mean()}',
          )