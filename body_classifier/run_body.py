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
import torchvision.models
import torch
import torchvision.transforms as tf
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datetime import datetime
import umap
import seaborn as sns
import csv

# MODEL AND DATALOADER PARAMETERS
Learning_Rate=0.001
width=height=224 # image width and height
batchSize=32
numWorkers=10
# %%

class CustomImageDataset(Dataset):
    def __init__(self, df, img_path = 'file_name', transforms=None):
        super().__init__()
        self.df = df
        self.img_path = img_path
        self.transforms = transforms
        self.label_dict = {"face": 0, "scalp": 1, "neck": 2, "shoulders": 3, "arms": 4, "hands": 5, "chest and abdomen": 6, "back": 7, "legs": 8, "genital and perianal": 9, "feet": 10, "dermoscope": 11}

    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        img_location = self.df[self.img_path].iloc[index]

        try:
            image = Image.open(img_location).convert('RGB')

        except:
            # choose another random image to load instead
            random_idx = np.random.choice(self.df.shape[0])
            with open('none_importable_images.txt','a+') as fh:
                fh.write(img_location +',' + self.df[self.img_path].iloc[random_idx] + '\n')
            return self.__getitem__(random_idx)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, img_location
        

data_transforms = {
    'test': tf.Compose([
        tf.Resize((height,width)),
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def build_df(csv_path, img_path):
    df = pd.read_csv(csv_path, header=0)
    df['file_name'] = img_path + '/' + df['file_name']
    return df

# %%

#df_all = build_df('/share/pi/ogevaert/zhang/SkinSegmentation/body_parts/body_train/image_path_results_final.csv',
                    #'/share/pi/ogevaert/sadee/skin/clinical/som-dermatology-photos-2020')
df_all = build_df('/share/pi/ogevaert/zhang/body_classifier/fitz17k_labeled.csv',
                    '/share/pi/ogevaert/sadee/skin/fitz17k/images')
test_length = len(df_all)
test_dataset = CustomImageDataset(df_all, transforms = data_transforms['test'])
test_dataloader = DataLoader(test_dataset, batch_size=batchSize,shuffle=False)

# %%

#modelPath = '/share/pi/ogevaert/zhang/SkinSegmentation/models/body_0613_allweighted/body_model_best_20230614_051547_26'
#modelPath = '/share/pi/ogevaert/zhang/SkinSegmentation/models/body_0613_allweighted/body_model_best_20230614_051843_18'
#modelPath = '/share/pi/ogevaert/zhang/SkinSegmentation/models/body_0613_allweighted/body_model_20230614_051547_49'
modelPath = '/share/pi/ogevaert/zhang/SkinSegmentation/models/body_0613_allweighted/body_model_20230614_051843_49'
#modelPath = '/share/pi/ogevaert/zhang/SkinSegmentation/models/body_0614_undersample/body_model_best_20230617_063559_11'
#modelPath = '/share/pi/ogevaert/zhang/SkinSegmentation/models/body_0614_undersample/body_model_20230617_063559_49'
#modelPath = '/share/pi/ogevaert/zhang/SkinSegmentation/models/body_0617_noboth/body_model_best_20230618_000024_8'
#modelPath = '/share/pi/ogevaert/zhang/SkinSegmentation/models/body_0617_noboth/body_model_20230618_000024_49'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, out_features=12, bias=True)
model = model.to(device)  # Set model to GPU or CPU
model.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu'))) # Load trained model
loss_fn = torch.nn.CrossEntropyLoss()
model.eval()

# %%

def run(model):
    model = model.to(device)
    model.eval()

    filename = '/share/pi/ogevaert/zhang/body_classifier/fitz17k_noundersample.csv'

    # create the csv file if it doesn't exist
    f = open(filename, 'w')
    writer = csv.writer(f)
    writer.writerow(['image_path', 'prediction', 'probabilities'])

    with torch.no_grad():
        for i, (inputs, paths) in enumerate(tqdm(test_dataloader)):

            inputs = inputs.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs,1)
            _, preds = torch.max(outputs, 1)
            
            # save the paths and the predictions in a csv file

            for i in range(len(paths)):
                preds_list = preds.detach().cpu().numpy().tolist()
                probs_list = probs.detach().cpu().numpy().tolist()
                writer.writerow([paths[i], preds_list[i], probs_list[i]])
    
    f.close()

run(model)