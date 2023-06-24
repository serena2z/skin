# %%
import pandas as pd 
import torch
import matplotlib.pyplot as plt
#from shape import shape_to_mask
from PIL import Image
import os
import glob as glob
import numpy as np
import cv2
import torchvision.models.segmentation
import torch
import torchvision.transforms as tf
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight

# MODEL AND DATALOADER PARAMETERS
Learning_Rate=0.001
# decrease the learning rate after 25 epochs- learning rate decay
width=height=224 # image width and height
# keep original size for resnet
batchSize=32
# increase to 64
numWorkers=10
# increase
# %%
class CustomImageDataset(Dataset):
    def __init__(self, df, img_path = 'image_file', label_col = 'label', transforms=None):
        super().__init__()
        self.df = df
        self.label_col = label_col
        self.img_path = img_path
        self.transforms = transforms
        #self.label_dict = {"face": 0, "scalp": 1, "ear": 2, "neck": 3, "shoulders": 4, "arms_upper": 5, "arms_lower": 6, "hands": 7, "chest": 8, "abdomen": 9, "back_upper": 10, "back_lower": 11, "hips_and_glutes": 12, "genital_and_perianal": 13, "legs_upper": 14, "legs_lower": 15, "feet": 16, "closeup": 17, "dermoscope": 18, "non-skin": 19 }
        self.label_dict = {"face": 0, "scalp": 1, "neck": 2, "arms": 3, "hands": 4, "chest and abdomen": 5, "back": 6, "legs": 7, "genital and perianal": 8, "feet": 9, "dermoscope": 10}


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        img_location = self.df[self.img_path].iloc[index]
        label = self.label_dict[self.df[self.label_col].iloc[index]]

        try:
            #print('img_location: ', img_location)
            image = Image.open(img_location).convert('RGB')
            # print image type
            #print(type(' image_type: ', image, '\n'))

        except:
            # choose another random image to load instead
            random_idx = np.random.choice(self.df.shape[0])
            print('img not found: ', img_location + '\n')
            #with open('none_importable_body_images.txt','a+') as fh:
                #fh.write(img_location +',' + self.df[self.img_path].iloc[random_idx] + '\n')
            return self.__getitem__(random_idx)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label

    def classes(self):
        return self.df[self.label_col].unique().tolist()
    
    def all_labels(self):
        return self.df[self.label_col].tolist()
        

# affine transformations
# vertical, jitter, horizontal flip, rotation, zoom, shear, etc
data_transforms = {
    'train': tf.Compose([
        tf.Resize((height,width)),
        tf.RandomHorizontalFlip(),
        tf.RandomVerticalFlip(),
        # tf.RandomRotation(30),
        # tf.RandomAffine(0, shear=10),
        # tf.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': tf.Compose([
        tf.Resize((height,width)),
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# %%
# def build_df(csv_path, img_path):
#     df = pd.read_csv(csv_path, header=0)
#     df['image_file'] = img_path + '/' + df['image_file']
#     return df

# df_train = build_df('/share/pi/ogevaert/zhang/body_classifier/body_train_45k_combined.csv',
#                     '/share/pi/ogevaert/sadee/skin/clinical/som-dermatology-photos-2020')

# df_val = build_df('/share/pi/ogevaert/zhang/body_classifier/body_val_45k_combined.csv',
#                     '/share/pi/ogevaert/sadee/skin/clinical/som-dermatology-photos-2020')

df_train = pd.read_csv('/share/pi/ogevaert/zhang/body_classifier/body_train_45k_combined_fitz.csv', header=0)
df_val = pd.read_csv('/share/pi/ogevaert/zhang/body_classifier/body_val_45k_combined_fitz.csv', header=0)

# %%
image_datasets = {'train':  CustomImageDataset(df_train,label_col = 'body_label', transforms = data_transforms['train']),
                  'val':    CustomImageDataset(df_val,label_col = 'body_label', transforms = data_transforms['val'])}

# WEIGHTED SAMPLER
labels = image_datasets['train'].all_labels()
label_dict = {"face": 0, "scalp": 1, "neck": 2, "arms": 3, "hands": 4, "chest and abdomen": 5, "back": 6, "legs": 7, "genital and perianal": 8, "feet": 9, "dermoscope": 10}
labels = [label_dict[label] for label in labels]
class_counts = torch.bincount(torch.tensor(labels))
weights = 1.0 / class_counts
sample_weights = [weights[label] for label in labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True) # type: ignore

# SAMPLING TOP 3
# class_counts = torch.bincount(torch.tensor(labels))
# # Get the top 3 classes with the highest counts
# top_classes = torch.argsort(class_counts, descending=True)[:3]
# # Calculate weights for each sample
# weights = 1.0 / class_counts
# # Assign higher weights to samples from the top 3 classes, and 1.0 to the rest
# sample_weights = torch.ones(len(labels))
# for class_idx in top_classes:
#     class_samples = torch.nonzero(torch.tensor(labels) == class_idx.item()).flatten()
#     sample_weights[class_samples] = weights[class_idx]
# print(sample_weights)
# # Create a WeightedRandomSampler with the sample weights
# sampler = WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)


dataloaders = {}
dataloaders['train'] = DataLoader(image_datasets['train'], batch_size=batchSize, sampler = sampler, num_workers=numWorkers)
dataloaders['val'] = DataLoader(image_datasets['val'], batch_size=batchSize, shuffle = True, num_workers=numWorkers)

# NO SAMPLING
# dataloaders = {x: DataLoader(image_datasets[x], batch_size=batchSize, shuffle=False, num_workers=numWorkers) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# %%
#--------------Load and set net and optimizer-------------------------------------
#print(torch.cuda.memory_summary(device=None, abbreviated=False))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# load resnet34 model
model = torchvision.models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, out_features=18, bias=True)
model.fc = torch.nn.Linear(num_ftrs, out_features=11, bias=True)

#freeze the first 3 layers of the model
# ct = 0
# for child in model.children():
#     ct += 1
#     if ct < 3:
#         for param in child.parameters():
#             param.requires_grad = False
        
    
model=model.to(device)
optimizer=torch.optim.Adam(params=model.parameters(),lr=Learning_Rate) # Create adam optimizer
# add gamma parameter for learning rate decay - 25 epochs, gamma=0.1
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
# class_w = image_datasets['train'].classes()
# all_labels = image_datasets['train'].all_labels()
# class_weights = compute_class_weight(class_weight='balanced', classes=class_w,  y=all_labels)
# class_weights = torch.FloatTensor(class_weights).cuda()
loss_fn = torch.nn.CrossEntropyLoss()
#loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights) # Create loss function

# %%
def train_one_epoch(epoch_index):
   running_loss = 0.
   running_corrects = 0.
   for i, data in enumerate(tqdm(dataloaders['train'])):
        # Every data instance is an input + label pair
      images, labels = data
      images = images.to(device)
      labels = labels.to(device)

      # Zero your gradients for every batch!
      optimizer.zero_grad()
      # Make predictions for this batch
      outputs = model(images)
      _, preds = torch.max(outputs, 1)
      # Compute the loss and its gradients
      loss = loss_fn(outputs, labels)
      #loss.requires_grad = True
      loss.backward()
      # Adjust learning weights
      optimizer.step()
      # Gather data and report per 100 batches
      running_loss += loss.detach().item() * images.size(0)
      running_corrects += torch.sum(preds == labels.data).detach()
      # if i % 100 == 99:
      #    last_loss = running_loss / 100 # loss per batch
      #    print('  batch {} loss: {}'.format(i + 1, last_loss))
      #    running_loss = 0.
   
   # update the scheduler
   scheduler.step()
   # calculate loss for epoch
   avg_loss = running_loss / dataset_sizes['train']
   avg_correct = running_corrects / dataset_sizes['train']
   return avg_loss, avg_correct

# %%

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
EPOCHS = 50
epoch_number = 0

highest_acc = 0.
best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch_number + 1))
    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss, avg_corrects = train_one_epoch(epoch_number)
    # We don't need gradients on to do reporting
    model.train(False)

    running_vloss = 0.0
    running_vcorrects = 0.
    with torch.no_grad():
        for i, vdata in enumerate(tqdm(dataloaders['val'])):
            vimages, vlabels = vdata
            vimages = vimages.to(device)
            vlabels = vlabels.to(device)
            voutputs = model(vimages)
            _, vpreds = torch.max(voutputs, 1)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss.detach().item() * vimages.size(0)
            running_vcorrects += torch.sum(vpreds == vlabels.data).detach()

    avg_vloss = running_vloss / dataset_sizes['val']
    avg_vcorrects = running_vcorrects / dataset_sizes['val']
    print('LOSS train {} train acc {} valid {} valid acc {}'.format(avg_loss, avg_corrects, avg_vloss, avg_vcorrects))
    
    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = '/share/pi/ogevaert/zhang/SkinSegmentation/models/body_0622/body_model_best_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    # save model every 10 epochs
    if (epoch_number+1) % 10 == 0:
        model_path = '/share/pi/ogevaert/zhang/SkinSegmentation/models/body_0622/body_model_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    # save the highest accuracy model
    if avg_vcorrects > highest_acc:
        highest_acc = avg_vcorrects
        model_path = '/share/pi/ogevaert/zhang/SkinSegmentation/models/body_0622/body_model_highestacc_{}_{}'.format(timestamp, epoch_number)
        torch.save(model.state_dict(), model_path)

    epoch_number += 1