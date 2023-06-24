import os
from PIL import Image
import numpy as np

folder = '/share/pi/ogevaert/zhang/all_clinical_masks'
save_folder = '/share/pi/ogevaert/zhang/all_somclinical_masks'

# if the save folder does not exist, create it
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# for every mask in the folder, change the pixel values to 0 and 255
for file in os.listdir(folder):
    # read the image
    image = Image.open(os.path.join(folder, file)).convert('L')
    # multiply each pixel by 255
    image = np.array(image) * 255
    print(file, np.unique(image))
    # convert the image to uint8
    image = image.astype(np.uint8)
    # save the image
    Image.fromarray(image).save(os.path.join(save_folder, file))
