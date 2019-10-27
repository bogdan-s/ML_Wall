import torch.optim as optim
from sklearn.metrics import roc_auc_score, f1_score
# from model import createDeepLabv3
from trainer import train_model
import datahandler

import os
import torch


 #             MODEL.PY
""" DeepLabv3 Model download and change the head for your prediction"""
from models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models


def createDeepLabv3(outputchannels=1):
    model = models.segmentation.deeplabv3_resnet101(
        pretrained=True, progress=True)
    # Added a Sigmoid activation after the last convolution layer
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    model.train()
    return model



# Command line arguments 
# parser = argparse.ArgumentParser()
# parser.add_argument(
#     "data_directory", help='Specify the dataset directory path')
# parser.add_argument(
#     "exp_directory", help='Specify the experiment directory where metrics and model weights shall be stored.')
# parser.add_argument("--epochs", default=25, type=int)
# parser.add_argument("--batchsize", default=4, type=int)

# args = parser.parse_args()


bpath = "D:/Python/ML_Wall/Save"     #Specify the experiment directory where metrics and model weights shall be stored.
data_dir = 'D:/Python/DataSets/ADE20K_Filtered/'  #Specify the dataset directory path
epochs = 25
batchsize = 4
# Create the deeplabv3 resnet101 model which is pretrained on a subset of COCO train2017, on the 20 categories that are present in the Pascal VOC dataset.
model = createDeepLabv3()
model.train()
# Create the experiment directory if not present
if not os.path.isdir(bpath):
    os.mkdir(bpath)


# Specify the loss function
criterion = torch.nn.MSELoss(reduction='mean')
# Specify the optimizer with a lower learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Specify the evalutation metrics
metrics = {'f1_score': f1_score, 'auroc': roc_auc_score}


# Create the dataloader
dataloaders = datahandler.get_dataloader_sep_folder(data_dir, batch_size=batchsize)
trained_model = train_model(model, criterion, dataloaders,
                            optimizer, bpath=bpath, metrics=metrics, num_epochs=epochs)


# Save the trained model
# torch.save({'model_state_dict':trained_model.state_dict()},os.path.join(bpath,'weights'))
torch.save(model, os.path.join(bpath, 'weights.pt'))    