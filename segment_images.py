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
from torch.utils.data import DataLoader
import datetime
from datetime import datetime
import random
import torchvision.transforms.functional as F

# %%
# MODEL AND DATALOADER PARAMETERS
Learning_Rate=1e-5
width=height=900 # image width and height
# %%
class CustomImageDataset(Dataset):
    def __init__(self, df, img_path = 'file_name', label_col = 'image_type_enc', transforms=False):
        super().__init__()
        self.df = df
        #self.label_col = label_col
        self.img_path = img_path
        self.transforms = transforms

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        img_location = self.df[self.img_path].iloc[index]
        #label = self.df[self.label_col].iloc[index]

        try:
            image = Image.open(img_location).convert('RGB')

        except:
            random_idx = np.random.choice(self.df.shape[0])
            # why do you need to pick a random index?            
            with open('none_importable_images.txt','a+') as fh:
                fh.write(img_location +', ' + self.df[self.img_path].iloc[random_idx] + '\n')                
            return self.__getitem__(random_idx)
        
        image_width, image_height = image.size

        if self.transforms is True:
            image = self.transform(image)
        else: 
            image = tf.Resize((height,width))(image)
            image = tf.ToTensor()(image)
            image = tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image) 

        return image, image_height, image_width, img_location

    #def classes(self):
        #return self.df[self.label_col].unique().tolist()

    def transform(self, image):
        # RESIZE
        image = tf.Resize((height,width))(image)

        # RANDOM HORIZONTAL FLIP
        if random.random() > 0.5:
            image = F.hflip(image)
        
        # RANDOM VERTICAL FLIP
        if random.random() > 0.5:
            image = F.vflip(image)

        # RANDOM AFFINE - ROTATION, TRANSLATION, SCALE, SHEAR
        if random.random() > 0.3:
            degree, translate, scale, shear = tf.RandomAffine.get_params(degrees=[-30, 30], translate=[0.3, 0.3], scale_ranges=[0.75, 1.25], shears=[-10, 10], img_size=(height, width))
            image = F.affine(image, degree, translate, scale, shear, interpolation=tf.InterpolationMode.BILINEAR)

        # COLOR JITTER
        order, brightness, contrast, saturation, hue = tf.ColorJitter.get_params(brightness=[0.9, 1.1], contrast=[0.9, 1.1], saturation=[0.9, 1.1], hue=[-0.1, 0.1])
        # apply the color jitter in the order specified
        for i in order:
            if i == 0:
                image = F.adjust_brightness(image, brightness)
            elif i == 1:
                image = F.adjust_contrast(image, contrast)
            elif i == 2:
                image = F.adjust_saturation(image, saturation)
            elif i == 3:
                image = F.adjust_hue(image, hue)

        # TO TENSOR
        image = tf.ToTensor()(image)

        # NORMALIZE
        image = tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)        

        return image
# %%
def build_df(csv_path, img_path):
    df = pd.read_csv(csv_path, header=0)
    df['file_name'] = img_path + '/' + df['file_name']
    return df
# %%
def run(dataloader, model_name, save_dir="./masks"):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
    model = model.to(device)  # Set net to GPU or CPU
    model.load_state_dict(torch.load(model_name, map_location=device)) # Load trained model
    loss_fn = torch.nn.CrossEntropyLoss()
    model.eval() # Set to evaluation mode

    for i, (image, height, width, img_file) in enumerate(tqdm(dataloader)):
        image = image.to(device)
        with torch.no_grad():
            Prd = model(image)['out']
        #resize all images in the batch size one by one to their orignal size
        for j in range(len(Prd)):
            temp = tf.Resize((int(height[j]),int(width[j])))(Prd[j])
            seg = torch.argmax(temp, 0).cpu().detach().numpy()
            # make the mask from 0 to 255
            seg = seg.astype(np.uint8) * 255
            # save image to masks using PIL as type 'L' and using the original image name with .PNG
            Image.fromarray(seg.astype(np.uint8)).convert('L').save(os.path.join(save_dir, os.path.basename(img_file[j]).split('.')[0] + '.png'))
# %%
# DEMO
batchSize=3
numWorkers=3
image_df = pd.read_csv('/share/pi/ogevaert/zhang/20230617_DDI/ddi_metadata.csv')
image_df['DDI_file'] = '/share/pi/ogevaert/zhang/20230617_DDI/' + image_df['DDI_file']
dataset = CustomImageDataset(image_df, img_path = 'DDI_file')
dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=False, num_workers=numWorkers)
# if the folder to save masks does not exist, create it
if not os.path.exists("/share/pi/ogevaert/zhang/all_DDI_masks"):
    os.makedirs("/share/pi/ogevaert/zhang/all_DDI_masks")
run(dataloader, model_name="/share/pi/ogevaert/zhang/SkinSegmentation/models/skin/model_best_20230413_202817_149", save_dir="/share/pi/ogevaert/zhang/all_DDI_masks")
