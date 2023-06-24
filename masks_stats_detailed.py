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
    
def get_center_of_mass(mask_image):
    center_of_mass = ndimage.center_of_mass(mask_image)
    center_of_mass_x = center_of_mass[1]
    center_of_mass_y = center_of_mass[0]
    return center_of_mass_x, center_of_mass_y

def get_pixel_count(mask_image):
    pixels = list(mask_image.getdata())
    count_0 = pixels.count(0)
    count_255 = pixels.count(255)
    return count_0, count_255

def get_contour_count(mask_image):
    contours, _ = cv2.findContours(mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    count = len(contours)
    #find the centroid of each contour
    centers = []
    sizes = []
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] > 0:
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])
            centers.append((centroid_x, centroid_y))
        
            # find the size of each contour
            sizes.append(cv2.contourArea(contour))

    return count, contours, centers, sizes

# %%
def build_df(csv_path, img_path):
    df = pd.read_csv(csv_path, header=0)
    df['file_name'] = img_path + '/' + df['file_name']
    return df
# %%
def run(dataframe, save_dir="./masks"):
    file_saved = False
    new_csv = pd.DataFrame(columns=['file_name', 'count_0', 'count_255', 'center_of_mass_x', 'center_of_mass_y', 'count', 'centers', 'sizes'])
    for i in tqdm(range(len(dataframe))):
        img_location = dataframe.iloc[i]['image_path']
        # loop over the batch
        image = Image.open(img_location).convert('L')
        # get pixel count
        count_0, count_255 = get_pixel_count(image)

        # get center of mass
        mask_image = cv2.imread(img_location, 0)  # Load as grayscale image
        center_of_mass_x, center_of_mass_y = get_center_of_mass(mask_image)

        # get contour count
        count, contours, centers, sizes = get_contour_count(mask_image)

        image_name = img_location.split('/')[-1]

        new_csv = new_csv.append({'file_name': image_name, 'count_0': count_0, 'count_255': count_255, 'center_of_mass_x': center_of_mass_x, 'center_of_mass_y': center_of_mass_y, 'count': count, 'centers': centers, 'sizes': sizes}, ignore_index=True)
    try:
        file_saved = True
        new_csv.to_csv(save_dir, index=False)
    except:
        print("Error saving csv file")
    return new_csv, file_saved

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

            # get the name of the sub_directory
            sub_folder_name = sub_folder.split('/')[-1]
            new_csv, file_saved = run(image_df, save_dir=folder + '/' + sub_folder_name + "_detailed_stats.csv")
            if file_saved == False:
                # save the dataframe as a csv file
                print('/share/pi/ogevaert/zhang/' + sub_folder_name + '_detailed_stats.csv')
                new_csv.to_csv('/share/pi/ogevaert/zhang/' + sub_folder_name + '_detailed_stats.csv', index=False)


# %%

if __name__ == "__main__":
    #folder_paths = ["/share/pi/ogevaert/sadee/skin/esteva", "/share/pi/ogevaert/sadee/skin/DermNet", "/share/pi/ogevaert/sadee/skin/DDI", "/share/pi/ogevaert/sadee/skin/clinical"]
    folder_paths = ["/share/pi/ogevaert/sadee/skin/DDI"]
    collate_folders(folder_paths)
