import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from CustomDataset import *
from helpers import *

from engine import train_one_epoch, evaluate
import utils
import transforms as T

import cv2

dataset = PennFudanDataset('data/')

# instantiate - dataset class and the data transforms

# use our dataset and defined transformations
dataset = PennFudanDataset('data', get_transform(train=True))
dataset_test = PennFudanDataset('data', get_transform(train=False))
print(len(dataset))
print(len(dataset_test))

# set percentage of split
train_per = 0.7
valid_per = 0.2
test_per  = 0.1

train_end = int(train_per*len(dataset)) + 1
valid_end = train_end + int(valid_per*len(dataset))
test_end  = valid_end + int(test_per*len(dataset))

# split the dataset in train and test set
torch.manual_seed(1)
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:train_end])
dataset_valid = torch.utils.data.Subset(dataset_test, indices[train_end:valid_end])
dataset_test = torch.utils.data.Subset(dataset_test, indices[valid_end:])

print(len(dataset))
print(len(dataset_valid))
print(len(dataset_test))

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_valid = torch.utils.data.DataLoader(
    dataset_valid, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

#instantiate - model and the optimizer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# our dataset has two classes only - background and person
num_classes = 2

# get the model using our helper function
model = get_instance_segmentation_model(num_classes)
# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

# let's train it for 10 epochs
num_epochs = 10

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # update the learning rate
    lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

model.eval()
with torch.no_grad():
  evaluate(model, data_loader_test, device=device)

for i in range(len(dataset_test)):
  print(i)
  img, _ = dataset_test[i]
  model.eval()
  with torch.no_grad():
    prediction = model([img.to(device)])
  img = prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy()
  name = 'img_' + str(i) + '.tif'
  cv2.imwrite("output/" +name, img)