from __future__ import print_function, division
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import ImageFile,Image
import  glob
ImageFile.LOAD_TRUNCATED_IMAGES = True
#plt.ion()

# Data augmentation and normalization for training
# Just normalization for validation




def predictions(model, image):
    model.eval()
    img = image.to(device)

    with torch.no_grad():
        outputs = model(img)
        outputs=torch.softmax(outputs, dim=1, dtype=float)
        _, preds = torch.max(outputs, 1)
    return class_names[preds]


def image_loader(loader, image):
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((64, 64)),
        #transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((64, 64)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data_dir = 'arranged_data_final'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=416,
                                             shuffle=True, num_workers=16)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
print(len(class_names))

with open('labels.txt', 'w') as filehandle:
    for listitem in class_names:
        filehandle.write('%s\n' % listitem)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

model_ft = model_ft.to(device)

model_ft.load_state_dict(torch.load('./weights/resnet50_f.pth'))

classes=glob.glob('./arranged_data_final/val/*')
all_images=[glob.glob(classes[i]+'/*') for i in range(len(classes)) ]
merged_images=list(itertools.chain.from_iterable(all_images))

import time
start_time=time.time()
count=0
total_correct=0
for j in merged_images:
    print(count)
    image=Image.open(j)
    image=image_loader(data_transforms['val'], image)
    pred_name=predictions(model_ft, image)
    if "_".join(pred_name.split()) in j.split('/'):
        total_correct+=1
    count=count+1


end_time=time.time()

print('Total time=', end_time-start_time)
print('Total images processed=', count)
print('Frames Per Seconds with Pytorch Model (JETSON NANO) =', count/(end_time-start_time))
print('Test Data accuracy with Pytorch Model (JETSON NANO) =', (total_correct/count)*100)
