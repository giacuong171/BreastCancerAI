import torch
import torch.nn as nn
import torch.nn.functional as F

#Class of a neural network should be inherited from nn.Module ( original class ) 
# by using super() function
def cal_output_size(input_size,kernel_size,stride,padding):
    return int((input_size - kernel_size + 2*padding)/stride) + 1

class CNN(nn.Module):
    def __init__(
            self,
            input_channels,
            channels,
            conv_kernels,
            padding,
            strides,
            pool_kernels,
            device='cpu'
    ):
        super(CNN,self).__init__()
        self.input_channels = input_channels
        self.channels = channels
        self.conv_kernels = conv_kernels
        self.padding = padding
        self.strides = strides
        self.pool_kernels = pool_kernels
        self.device = device
        cnn = nn.Sequential()

        def add_layer(i, batch_norm=False,leaky_relu=False):
            n_in = self.input_channels if i == 0 else self.channels[i - 1]
            n_out = self.channels[i]
            cnn.add_module(
                f'conv{i + 1}',nn.Conv2d(n_in,n_out,self.conv_kernels[i],self.strides[i],self.padding[i])
            )
            if batch_norm:
                cnn.add_module(f'batch_norm{i + 1}',nn.BatchNorm2d(n_out))
            if leaky_relu:
                cnn.add_module(f'leaky_relu{i + 1}',nn.LeakyReLU(inplace=True))
            else:
                cnn.add_module(f'relu{i + 1}',nn.ReLU(inplace=True))

            cnn.add_module(f'pool{i + 1}',nn.MaxPool2d(self.pool_kernels[i]))
        
        add_layer(0)
        add_layer(1)
        add_layer(2)
        
        #Connecting the CNN to the fully connected layer
        cnn.add_module('flatten',nn.Flatten()) 
        
        #dim=2 when the label is OneHotEncoded or 3D data
        cnn.add_module('softmax',nn.Softmax(dim=1))
        self.device = device
        self.cnn = cnn.to(device)

    def forward(self,x):
        x=self.cnn(x)
        return x
        

    