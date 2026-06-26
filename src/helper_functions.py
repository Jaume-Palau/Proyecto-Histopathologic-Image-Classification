import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataset import DatasetWrapper_Train

import torch
import torch.nn as nn

################################################################################
## HELPER FUNCTION - Plot the training and validation loss for each epoch
################################################################################

def plot_results(config, training_loss, validation_loss, filename_path):
    """Helper Function: Plot and save the train/validation loss average over epochs."""
    print_textbox = False
    if print_textbox: textbox_text = "\n".join(  "=".join((key, str(value))) for (key, value) in config.items()  )
    
    training_loss = np.mean(training_loss, axis=1)
    validation_loss = np.mean(validation_loss, axis=1)
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(training_loss, 
            linestyle="dashed", 
            linewidth=2, 
            alpha=0.5,
            label="Training Loss (avg per epoch)")
    ax.scatter(x=range(len(training_loss)),
               y=training_loss)
    ax.plot(validation_loss, 
            linestyle="dashed", 
            linewidth=2, 
            alpha=0.5,
            label="Validation Loss (avg per epoch)")
    ax.scatter(x=range(len(validation_loss)), 
               y=validation_loss)

    ax.set_title(f"Training/Validation Loss of Trial: {config['trial_name']}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    if print_textbox: 
        ax.text(0.9, 0.1, textbox_text, fontsize=12, 
            transform=ax.transAxes,
            verticalalignment="bottom",
            horizontalalignment="right", 
            bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    ax.legend()
    
    fig.show()
    fig.savefig(Path(filename_path))
    return

################################################################################
## HELPER FUNCTION - Calculate conv layer input and outputs dimensions
################################################################################

def calc_dim(height:int, width:int, conv_layer:torch.nn):
    """Calculate the output dimensions for convolutional layers."""
    
    ## Case checking - Sometimes the values can be returned as int instead of a tuple
    kernel_h, kernel_w = (
        (conv_layer.kernel_size, conv_layer.kernel_size) if isinstance(conv_layer.kernel_size, int) 
        else conv_layer.kernel_size
    )
    
    padding_h, padding_w = (
        (conv_layer.padding, conv_layer.padding) if isinstance(conv_layer.padding, int) 
        else conv_layer.padding
    )
    
    stride_h, stride_w = (
        (conv_layer.stride, conv_layer.stride) if isinstance(conv_layer.stride, int) 
        else conv_layer.stride
    )
    
    dilation_h, dilation_w = (
        (conv_layer.dilation, conv_layer.dilation) if isinstance(conv_layer.dilation, int) 
        else conv_layer.dilation
    )
    
    ## Calculate the output dimensions - Src: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
    output_height = int( (height + 2*padding_h - dilation_h*(kernel_h - 1) - 1)/stride_h + 1 )
    output_width = int( (width + 2*padding_w - dilation_w*(kernel_w-1) - 1)/stride_w +1 )
    
    return (output_height, output_width)


################################################################################
## Analyze the balance of labels
################################################################################

def plot_label_balance(labels):
    
    ## Get an instance of the dataloader
    training_set = DatasetWrapper_Train()

    ## Count the number of cancer / no-cancer labels
    count_no_cancer = np.count_nonzero(np.equal(training_set.labels, 0))
    count_cancer = np.count_nonzero(np.equal(training_set.labels, 1))


    ## Plot bar chart of cancer vs no cancer 
    fig, ax = plt.subplots(figsize=(3, 5))
    barchart = ax.bar(
        ["no_cancer", "cancer"], [count_no_cancer, count_cancer], 
        color=["#249A41", "#E92E18"],
        width=0.3,
        align="center",
    )
    ax.bar_label(barchart, labels=[count_no_cancer, count_cancer], padding=1)
    ax.set_title("Training Set Label Counts")
    ax.text(
        0.3, 122000, f"no_cancer:cancer = {round(count_no_cancer/count_cancer, 3)}", 
        ha="left", va="center",
        wrap=True,
        bbox=dict(facecolor="#EFEFEF", 
                alpha=0.3)
    )
    fig.show()

    ## Check
    print("No-cancer count: ", count_no_cancer)
    print("Cancer count: ", count_cancer)



if  __name__ == "__main__":

    ## IGNORE - Tests
    test_conv_layer = nn.Conv2d(3, 3, 5, padding=1)
    assert (3, 3) == calc_dim(5, 5, test_conv_layer)