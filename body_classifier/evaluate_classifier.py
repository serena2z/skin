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
from datetime import datetime

# MODEL AND DATALOADER PARAMETERS
Learning_Rate=0.001
width=height=224 # image width and height
batchSize=64
numWorkers=15

# %%
#modelPath = '/share/pi/ogevaert/zhang/SkinSegmentation/models/body_0522/body_model_best_20230522_202456_10'
#modelPath = '/share/pi/ogevaert/zhang/SkinSegmentation/models/body_0527/body_model_20230528_014432_89'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, out_features=20, bias=True)
model = model.to(device)  # Set model to GPU or CPU
model.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu'))) # Load trained model
loss_fn = torch.nn.CrossEntropyLoss()
model.eval()
# set torch.no_grad to true    

# %%
class CustomImageDataset(Dataset):
    def __init__(self, df, img_path = 'image_file', label_col = 'label', transforms=None):
        super().__init__()
        self.df = df
        self.label_col = label_col
        self.img_path = img_path
        self.transforms = transforms
        #self.label_dict = {"inconclusive": 0, "face": 1, "scalp": 2, "ear": 3, "neck": 4, "shoulders": 5, "arms_upper": 6, "arms_lower": 7, "hands": 8, "chest": 9, "abdomen": 10, "back_upper": 11, "back_lower": 12, "hips_and_glutes": 13, "genital_and_perianal": 14, "legs_upper": 15, "legs_lower": 16, "feet": 17 }
        self.label_dict = {"face": 0, "scalp": 1, "ear": 2, "neck": 3, "shoulders": 4, "arms_upper": 5, "arms_lower": 6, "hands": 7, "chest": 8, "abdomen": 9, "back_upper": 10, "back_lower": 11, "hips_and_glutes": 12, "genital_and_perianal": 13, "legs_upper": 14, "legs_lower": 15, "feet": 16, "closeup": 17, "dermoscope": 18, "non-skin": 19 }

    
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
            with open('none_importable_images.txt','a+') as fh:
                fh.write(img_location +',' + self.df[self.img_path].iloc[random_idx] + '\n')
            return self.__getitem__(random_idx)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label, img_location

    def classes(self):
        return self.df[self.label_col].unique().tolist()
        

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
    df['image_file'] = img_path + '/' + df['image_file']
    return df

# %%

df_test = build_df('/share/pi/ogevaert/zhang/body_classifier/body_test_2.csv',
                    '/share/pi/ogevaert/sadee/skin/clinical/som-dermatology-photos-2020')
test_length = len(df_test)
test_dataset = CustomImageDataset(df_test,label_col = 'body_label', transforms = data_transforms['test'])
test_dataloader = DataLoader(test_dataset, batch_size=batchSize,shuffle=True, num_workers=numWorkers)
# %%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay

def evaluate(model):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dataloader)):
            images, labels, _ = data
            images = images.to(device)
            labels = labels.to(device)
            # set torch.no_grad to true
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
            running_corrects += torch.sum(preds == labels.data)
        avg_loss = running_loss / test_length
        avg_corrects = running_corrects.double() / test_length
        print('LOSS {} ACC {}'.format(avg_loss, avg_corrects))

def generate_predictions(model):
    model.eval()
    y_true = []
    y_pred = []
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dataloader)):
            images, labels, _ = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            # append each batch to the list
            y_true.append(labels)
            y_pred.append(preds)
    # concatenate the list of tensors into a single tensor
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        # copy the tensors to the CPU
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()
        # convert to numpy arrays
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()
        return y_true, y_pred
        
# %%
evaluate(model)
y_true, y_pred = generate_predictions(model)

# %%
label_dict = {"face": 0, "scalp": 1, "ear": 2, "neck": 3, "shoulders": 4, "arms_upper": 5, "arms_lower": 6, "hands": 7, "chest": 8, "abdomen": 9, "back_upper": 10, "back_lower": 11, "hips_and_glutes": 12, "genital_and_perianal": 13, "legs_upper": 14, "legs_lower": 15, "feet": 16, "closeup": 17, "dermoscope": 18, "non-skin": 19 }
labels = list(label_dict.keys())
cm = confusion_matrix(y_true, y_pred, normalize='true')
f1 = f1_score(y_true, y_pred, average=None)
disp = ConfusionMatrixDisplay(cm, display_labels=labels)
# rotate x-axis labels
disp.plot(xticks_rotation='vertical', values_format='.1f')
# round the numbers in the confusion matrix to 1 decimal place
# print the f1 score with the label
for i in range(len(labels)):
    print(labels[i], f1[i])

# make text smaller
plt.rcParams.update({'font.size': 7})

# save the confusion matrix
plt.savefig('/share/pi/ogevaert/zhang/body_classifier/confusion_matrix.png', dpi=300, bbox_inches='tight')