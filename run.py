from pathlib import Path


import torch
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision.transforms import Resize

from models.dataset import BreastCancerDataset
from models.model_admin import CNN
from models.training import train_model

def main():
    data_dir = Path("/media/jacky/data_4tb/data/BreaKHis_v1/histology_slides/breast")
    #Preprocessing the data
    transform = Resize((46,70))
    BCdata = BreastCancerDataset(data_dir,transform = transform)
    train_size = int(0.8 * len(BCdata))
    test_size = len(BCdata) - train_size
    train_set,test_set = random_split(BCdata,[train_size,test_size])
    train_dataloader = DataLoader(train_set,batch_size=32,shuffle=True)
    test_dataloader = DataLoader(test_set,batch_size=32,shuffle=True)
    print(BCdata[0][0].shape)
    ##############################
    #Parameters for model
    ##############################
    input_channels = BCdata[0][0].shape[0]
    channels = [32,64,128]
    conv_kernels= [3,3,3]
    padding = [1,1,1]
    strides = [1,1,1]
    pool_kernels = [2,2,2]
    model = CNN(input_channels,channels,conv_kernels,padding,strides,pool_kernels)

    best_model, best_val_f1, best_train_f1, val_f1s, train_f1s = train_model(model,train_dataloader,test_dataloader,num_epochs=10,device='cpu')
    torch.save(best_model.state_dict(), "best_model.pth")
    torch.save(model.state_dict(), "model.pth")
    print(f"Best validation f1: {best_val_f1}")
    print(f"Best train f1: {best_train_f1}")
    print(f"Validation f1s: {val_f1s}")
    print(f"Train f1s: {train_f1s}")



if __name__=="__main__":
    main()