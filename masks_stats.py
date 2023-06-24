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
from scipy import ndimage

# %%
# MODEL AND DATALOADER PARAMETERS
Learning_Rate=1e-5
width=height=900 # image width and height
# %%
class CustomImageDataset(Dataset):
    def __init__(self, df, img_path = 'file_name', transforms=None):
        super().__init__()
        self.df = df
        self.img_path = img_path
        self.transforms = transforms
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        img_location = self.df[self.img_path].iloc[index]

        try:
            image = Image.open(img_location).convert('L')
            # get pixel count
            count_0, count_255 = self.get_pixel_count(image)

            # get center of mass
            mask_image = cv2.imread(img_location, 0)  # Load as grayscale image
            center_of_mass_x, center_of_mass_y = self.get_center_of_mass(mask_image)

            # get contour count
            count = self.get_contour_count(mask_image)


        except:
            # choose another random image to load instead
            random_idx = np.random.choice(self.df.shape[0])
            with open('none_importable_images.txt','a+') as fh:
                fh.write(img_location +',' + self.df[self.img_path].iloc[random_idx] + '\n')
            return self.__getitem__(random_idx)

        if self.transforms is not None:
            image = self.transforms(image)

        return count_0, count_255, center_of_mass_x, center_of_mass_y, count, img_location
    
    def get_center_of_mass(self, mask_image):
        center_of_mass = ndimage.center_of_mass(mask_image)
        center_of_mass_x = center_of_mass[1]
        center_of_mass_y = center_of_mass[0]
        return center_of_mass_x, center_of_mass_y
    
    def get_pixel_count(self, mask_image):
        pixels = list(mask_image.getdata())
        count_0 = pixels.count(0)
        count_255 = pixels.count(255)
        return count_0, count_255
    
    def get_contour_count(self, mask_image):
        contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        count = len(contours)
        #find the centroid of each contour
        # centers = []
        # sizes = []
        # for contour in contours:
        #     M = cv2.moments(contour)
        #     if M['m00'] > 0:
        #         centroid_x = int(M['m10'] / M['m00'])
        #         centroid_y = int(M['m01'] / M['m00'])
        #         centers.append((centroid_x, centroid_y))
            
        #         # find the size of each contour
        #         sizes.append(cv2.contourArea(contour))

        return count
    

data_transforms = {
    'test': tf.Compose([
        tf.Resize((height,width)),
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}
# %%
def build_df(csv_path, img_path):
    df = pd.read_csv(csv_path, header=0)
    df['file_name'] = img_path + '/' + df['file_name']
    return df
# %%
def run(dataloader, save_dir="./masks"):
    new_csv = pd.DataFrame(columns=['file_name', 'count_0', 'count_255', 'center_of_mass_x', 'center_of_mass_y', 'count', 'centers', 'sizes'])
    for batch in tqdm(dataloader):
        count_0, count_255, center_of_mass_x, center_of_mass_y, count, img_location = batch
        # loop over the batch
        for i in range(len(img_location)):
            print('i: ', i, len(img_location))
            # save the image
            img_name = img_location[i]
            curr_count_0 = count_0[i].numpy()
            curr_count_255 = count_255[i].numpy()
            curr_center_of_mass_x = center_of_mass_x[i].numpy()
            curr_center_of_mass_y = center_of_mass_y[i].numpy()
            curr_count = count[i].numpy()
            # curr_centers = centers[i]
            # curr_sizes = sizes[i]

            new_csv = new_csv.append({'file_name': img_name, 'count_0': curr_count_0, 'count_255': curr_count_255, 'center_of_mass_x': curr_center_of_mass_x, 'center_of_mass_y': curr_center_of_mass_y, 'count': curr_count}, ignore_index=True)

    new_csv.to_csv(save_dir, index=False)

# %%
# DEMO
batchSize=32
numWorkers=14

def collate_folders(folder_paths):
    for folder in folder_paths:
        # get the path of all the folders that starts with "predicted"
        sub_folder_paths = glob.glob(folder + '/predicted*')
        # get the file called all_images.txt
        all_images_file = glob.glob(folder + '/all_images.txt')
        print(all_images_file)
        # build the dataloader for each folder
        for sub_folder in sub_folder_paths:
            # read all_images_file into a dataframe
            image_df = pd.read_fwf(all_images_file[0], header=0)
            image_df['image_path'] = image_df['image_path'].str.split('/').str[-1]
            image_df['image_path'] = sub_folder + '/' + image_df['image_path']
            image_df['image_path'] = image_df['image_path'].str.replace('.jpg', '.png')

            dataset = CustomImageDataset(image_df, img_path='image_path')
            dataloader = DataLoader(dataset, batch_size=batchSize, shuffle=False, num_workers=numWorkers)

            # get the name of the sub_directory
            sub_folder_name = sub_folder.split('/')[-1]
            run(dataloader, save_dir=folder + '/' + sub_folder_name + "_simple_stats.csv")

# %%

if __name__ == "__main__":
    folder_paths = ["/share/pi/ogevaert/sadee/skin/fitz17k", "/share/pi/ogevaert/sadee/skin/esteva", "/share/pi/ogevaert/sadee/skin/DermNet", "/share/pi/ogevaert/sadee/skin/DDI", "/share/pi/ogevaert/sadee/skin/clinical"]
    collate_folders(folder_paths)
