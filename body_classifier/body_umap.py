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

# MODEL AND DATALOADER PARAMETERS
Learning_Rate=0.001
width=height=224 # image width and height
batchSize=32
numWorkers=8
# %%

class CustomImageDataset(Dataset):
    def __init__(self, df, img_path = 'image_file', label_col = 'label', transforms=None):
        super().__init__()
        self.df = df
        self.label_col = label_col
        self.img_path = img_path
        self.transforms = transforms
        #self.label_dict = {"inconclusive": 0, "face": 1, "scalp": 2, "ear": 3, "neck": 4, "shoulders": 5, "arms_upper": 6, "arms_lower": 7, "hands": 8, "chest": 9, "abdomen": 10, "back_upper": 11, "back_lower": 12, "hips_and_glutes": 13, "genital_and_perianal": 14, "legs_upper": 15, "legs_lower": 16, "feet": 17 }
        #self.label_dict = {"face": 0, "scalp": 1, "ear": 2, "neck": 3, "shoulders": 4, "arms_upper": 5, "arms_lower": 6, "hands": 7, "chest": 8, "abdomen": 9, "back_upper": 10, "back_lower": 11, "hips_and_glutes": 12, "genital_and_perianal": 13, "legs_upper": 14, "legs_lower": 15, "feet": 16, "dermoscope": 17}
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

def build_df(csv_path, img_path):
    df = pd.read_csv(csv_path, header=0)
    df['image_file'] = img_path + '/' + df['image_file']
    return df

# %%

#df_all = build_df('/share/pi/ogevaert/zhang/SkinSegmentation/body_parts/body_train/image_path_results_final.csv',
                    #'/share/pi/ogevaert/sadee/skin/clinical/som-dermatology-photos-2020')
# df_all = build_df('/share/pi/ogevaert/zhang/body_classifier/image_path_results_final_2.csv',
#                     '/share/pi/ogevaert/sadee/skin/clinical/som-dermatology-photos-2020')
# df_all = build_df('/share/pi/ogevaert/zhang/body_classifier/body_test_45k.csv',
#                     '/share/pi/ogevaert/sadee/skin/clinical/som-dermatology-photos-2020')
df_all = pd.read_csv('/share/pi/ogevaert/zhang/body_classifier/body_test_45k_combined_fitz.csv', header=0)
test_length = len(df_all)
test_dataset = CustomImageDataset(df_all,label_col = 'body_label', transforms = data_transforms['test'])
test_dataloader = DataLoader(test_dataset, batch_size=batchSize,shuffle=True, num_workers=numWorkers)

# %%

#modelPath = '/share/pi/ogevaert/zhang/SkinSegmentation/models/body_0613_allweighted/body_model_best_20230614_051547_26'
#modelPath = '/share/pi/ogevaert/zhang/SkinSegmentation/models/body_0613_allweighted/body_model_best_20230614_051843_18'
#modelPath = '/share/pi/ogevaert/zhang/SkinSegmentation/models/body_0613_allweighted/body_model_20230614_051547_49'
#modelPath = '/share/pi/ogevaert/zhang/SkinSegmentation/models/body_0613_allweighted/body_model_20230614_051843_49'
#modelPath = '/share/pi/ogevaert/zhang/SkinSegmentation/models/body_0614_undersample/body_model_best_20230617_063559_11'
#modelPath = '/share/pi/ogevaert/zhang/SkinSegmentation/models/body_0614_undersample/body_model_20230617_063559_49'
#modelPath = '/share/pi/ogevaert/zhang/SkinSegmentation/models/body_0617_noboth/body_model_best_20230618_000024_8'
#modelPath = '/share/pi/ogevaert/zhang/SkinSegmentation/models/body_0617_noboth/body_model_20230618_000024_49'
modelPath = '/share/pi/ogevaert/zhang/SkinSegmentation/models/body_0622/body_model_highestacc_20230623_004350_36'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, out_features=11, bias=True)
model = model.to(device)  # Set model to GPU or CPU
model.load_state_dict(torch.load(modelPath, map_location=torch.device('cpu'))) # Load trained model
loss_fn = torch.nn.CrossEntropyLoss()
model.eval()

# %%

def assess_model_softmax_features(model,layer = -1):
    model = model.to(device)
    model.eval()
    path_lst = []
    label_lst = []
    pred_lst = []
    probs_lst = []
    feat_lst = []
    
    feature_extractor = torch.nn.Sequential(*list(model.children())[:layer])


    with torch.no_grad():
        for i, (inputs, labels, paths) in enumerate(tqdm(test_dataloader)):

            inputs = inputs.to(device)
            labels = labels.to(device)            

            outputs = model(inputs)
            probs = torch.softmax(outputs,1)
            _, preds = torch.max(outputs, 1)
            
            features_outputs = feature_extractor(inputs)

            #this is for batches
            for j in range(inputs.size()[0]):                
                path_lst.append(paths[j])
                label_lst.append(labels[j].to('cpu').tolist())
                pred_lst.append(preds[j].to('cpu').tolist())
                probs_lst.append(probs[j,:].to('cpu').tolist())
                feat_lst.append(features_outputs[j].to('cpu').numpy())
                
        return path_lst,label_lst,pred_lst,probs_lst,feat_lst
    
# %%

path,true,pred,prob,feat = assess_model_softmax_features(model)


#%%

last_hidden_layer_features = [f.flatten() for f in feat]
features = np.vstack(last_hidden_layer_features)



reducer = umap.UMAP(random_state=42)
reducer.fit(features)
embedding = reducer.transform(features)
umap_out_d2 = embedding
coord_1 = umap_out_d2[:,0]
coord_2 = umap_out_d2[:,1]
print(umap_out_d2.shape)

# save the above lists to a csv file
#df = pd.DataFrame({'path':path,'true':true,'pred':pred,'prob':prob, 'coord_1':coord_1, 'coord_2':coord_2})
#df.to_csv('/share/pi/ogevaert/zhang/body_classifier/body_softmax_features_16k.csv')
pd.set_option('display.max_colwidth', None)
np.set_printoptions(threshold=np.inf)
# make sure all of the list is displayed
last_hidden_layer_features = [np.array2string(f, separator=',') for f in last_hidden_layer_features]

df = pd.DataFrame({'path':path,'true':true,'pred':pred,'prob':prob, 'feat':last_hidden_layer_features, 'coord_1':coord_1, 'coord_2':coord_2})
df.to_csv('/share/pi/ogevaert/zhang/body_classifier/body_softmax_features_combined_with_fitz.csv')

# %%

import matplotlib.patches as mpatches

# graph umap_out_d2 matching the values to their labels
label_dict = {"face": 0, "scalp": 1, "neck": 2, "arms": 3, "hands": 4, "chest and abdomen": 5, "back": 6, "legs": 7, "genital and perianal": 8, "feet": 9, "dermoscope": 10}
opposite_dict = {v: k for k, v in label_dict.items()}
# map the labels to colors
# color_dict = {0: 'rosybrown', 1: 'brown', 2: 'maroon', 3: 'darkred', 4: 'salmon', 5: 'orangered', 6: 'chocolate', 7: 'sandybrown', 8: 'darkorange', 9: 'tan', 10: 'moccasin', 11: 'darkgoldenrod', 12: 'gold', 13: 'khaki', 14: 'darkkhaki', 15: 'olive', 16: 'yellowgreen', 17: 'black'}
color_dict = {0: 'darkorchid', 1: 'violet', 2: 'mediumvioletred', 3: 'red', 4: 'orange', 5: 'cornflowerblue', 6: 'blue', 7: 'green', 8: 'yellow', 9: 'lawngreen', 10: 'black'}
plt.scatter(umap_out_d2[:, 0], umap_out_d2[:, 1], c=[color_dict[i] for i in true], s=0.5, cmap='Spectral')
plt.legend(handles=[mpatches.Patch(color=color_dict[i], label=opposite_dict[i]) for i in range(12)], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.title('UMAP projection of the body parts dataset', fontsize=24)
plt.savefig('/share/pi/ogevaert/zhang/body_classifier/umap_body_parts_45k_combined_withfitz.png', dpi=300, bbox_inches='tight')
# %%