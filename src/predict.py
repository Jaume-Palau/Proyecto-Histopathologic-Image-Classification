from itertools import chain

import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from dataset import DatasetWrapper_Test
from model import CustomCNN
from model_more_spatial import CustomCNN_MoreSpatial
from src.config import *

def load_trained_model(checkpoint_path, model : CustomCNN | CustomCNN_MoreSpatial):
    ## Load the model state from output file
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loaded_checkpoint = torch.load(
        checkpoint_path,  
    map_location=torch.device(device)
    )
    ## Create model and load the model state
    model.load_state_dict(loaded_checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def predict_test_set(model: CustomCNN | CustomCNN_MoreSpatial, output_type: str ):

    """
    Generate predictions for the test set.

    Args:
        model: PyTorch model used for inference.
        output_type (str): Type of model output.
            - "log_probs": model returns log-probabilities, e.g. LogSoftmax + NLLLoss.
            - "logits": model returns raw logits, e.g. CrossEntropyLoss.
    """

    model.eval()

    ## Inference with model 1a and save results
    test_dataloader = torch.utils.data.DataLoader(  # Create a dataloader to make things easier
        dataset = DatasetWrapper_Test(),  # The competition test dataset
        batch_size = BATCH_SIZE, 
        shuffle = False, 
        num_workers = 10, 
        prefetch_factor = 5000,
        drop_last = False,
        pin_memory = True,  # Pin memory for faster transfer to GPU
    )

    file_id_holder = []  # Test set is small enough thus holding output in memory
    results_holder = np.zeros(shape=(len(test_dataloader), BATCH_SIZE) )   # Test set is small enough thus holding output in memory
    for idx, (file_ids, images) in tqdm(enumerate(test_dataloader), desc="Inferencing test set"):
        with torch.no_grad():  # No gradient mode

            assert len(file_ids) == len(images), "File IDs and images don't have the same number of records."
            
            images = images.to(DEVICE)  # Transfer to GPU for faster inferencing
            
            ## Model inference
            output = model(images)

            if output_type == "log_probs":
                probs = torch.exp(output)
                predictions = probs[:, 1]
                
            elif output_type == "logits":
                predictions = torch.softmax(output, dim=1)[:, 1]
            
            ## Append results to holder
            predictions = predictions.cpu().numpy()  # Need to copy to CPU first
            if len(predictions) != BATCH_SIZE:  # The last batch typically is not a full-sized batch
                predictions = np.pad(predictions, (0, BATCH_SIZE - len(predictions)), "constant", constant_values=(0))
            results_holder[idx, :] = predictions
            file_id_holder.append(file_ids)

    ## Reshape the 2D array to an 1D array
    file_id_holder_new = list(chain(*file_id_holder))  # Convert list of lists into just list
    results_holder_new = results_holder.reshape((-1))[:len(file_id_holder_new)].astype(float)  # Remove the extra padded elements

    return file_id_holder_new, results_holder_new


def create_submisssion(file_id_holder_new,results_holder_new,output_path=SUBMISSIONS_DIR):
    ## Save the data in CSV for submission
    df = pd.DataFrame(
        {"id":file_id_holder_new, 
        "label":results_holder_new},
        )
    df.to_csv(str(output_path), index=False)






if __name__ == "__main__":

    model_path = Path("outputs/models/custom-cnn-morespatial/best_model.pt")

    model = load_trained_model(
        checkpoint_path=model_path
    )

    names, predictions = predict_test_set(model)

    submision_path = Path(SUBMISSIONS_DIR/"submission3.csv")

    create_submisssion(
        file_id_holder_new= names,
        results_holder_new=predictions,
        output_path= submision_path
    )