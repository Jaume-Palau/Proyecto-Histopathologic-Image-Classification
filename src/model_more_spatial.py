import torch
import torch.nn as nn
import torchinfo

from helper_functions import calc_dim

'''

Cambios en la estructura de la red original:
- Cambio el tamaño de la primera capa densa de 100 a 32, para que tenga menos parámetros y sea más rápida.
- Elimino la ultima capa de MaxPooling, para que la red tenga más información espacial y pueda aprender mejor.
- Aumento el padding de la cuarta capa convolucional para mantener las dimensiones.

'''

class CustomCNN_MoreSpatial(nn.Module):
    def __init__(self, input_channels:int=3, input_height:int=46, input_width:int=46):
        super().__init__()  # nn.Module constructor
        
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        
        ## Create/Define the loss function
        self._loss_function = nn.NLLLoss
        
        ## Create all the layers for the network
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=input_channels, 
                               out_channels=8, 
                               kernel_size=3, 
                               padding=0)
        self.leak_relu1 = nn.LeakyReLU()  # Activation layer doesn't change the output dimensions
        out_h, out_w = calc_dim(input_height, input_width, self.conv1)
        self.maxpool1 = nn.MaxPool2d(2)
        out_h, out_w = calc_dim(out_h, out_w, self.maxpool1)
                               
        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels, 
                               out_channels=16, 
                               kernel_size=3, 
                               padding=0)
        self.leak_relu2 = nn.LeakyReLU()
        out_h, out_w = calc_dim(out_h, out_w, self.conv2)
        self.maxpool2 = nn.MaxPool2d(2)
        out_h, out_w = calc_dim(out_h, out_w, self.maxpool2)
        
       # Layer 3
        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels, 
                               out_channels=32, 
                               kernel_size=3, 
                               padding=0)
        self.leak_relu3 = nn.LeakyReLU()
        out_h, out_w = calc_dim(out_h, out_w, self.conv3)
        self.maxpool3 = nn.MaxPool2d(2)
        out_h, out_w = calc_dim(out_h, out_w, self.maxpool3)
        
       # Layer 4
        self.conv4 = nn.Conv2d(in_channels=self.conv3.out_channels, 
                               out_channels=64, 
                               kernel_size=3, 
                               padding=1)
        self.leak_relu4 = nn.LeakyReLU()
        out_h, out_w = calc_dim(out_h, out_w, self.conv4)
        # self.maxpool4 = nn.MaxPool2d(2)
        # out_h, out_w = calc_dim(out_h, out_w, self.maxpool4)

        # Classification Layer 1
        self.flattened_dim = out_h * out_w * self.conv4.out_channels
        self.dense1 = nn.Linear(in_features = self.flattened_dim, 
                                out_features = 32)
                               
        # Classification Layer 2
        self.dense2 = nn.Linear(in_features = 32, 
                                out_features = 2)  # Binary classification
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        """Override nn.Module forward method."""
        x = self.conv1(x)
        x = self.leak_relu1(x)
        x = self.maxpool1(x)
                               
        x = self.conv2(x)
        x = self.leak_relu2(x)
        x = self.maxpool2(x)
                               
        x = self.conv3(x)
        x = self.leak_relu3(x)
        x = self.maxpool3(x)
        
        x = self.conv4(x)
        x = self.leak_relu4(x)
        # x = self.maxpool4(x)

        x = x.view(-1, self.flattened_dim)  # Flatten the tensor                       
        x = self.dense1(x)
        x = self.leak_relu4(x)
        
        x = self.dense2(x)
        x = self.logsoftmax(x)
        
        return x
    
    def get_optimizer(self, **kwargs):
        """Return the optimizer used by this network."""
        return torch.optim.Adam(self.parameters(), **kwargs)
    
    def get_loss_function(self, **kwargs):
        """Return the loss function used by this network"""
        return self._loss_function(**kwargs)
    
    def show_summary(self):
        """Show the model summary information."""
        return torchinfo.summary(
            self, 
            input_size=(self.input_channels, self.input_height, self.input_width), 
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu").type,
            verbose=0)


if __name__ == "__main__":
        
    ## IGNORE - TEST
    temp_model = CustomCNN_MoreSpatial()
    print(temp_model.show_summary())
