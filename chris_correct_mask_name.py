# %%
# ------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import glob
from tqdm import tqdm

# %%

mask_files = glob.glob('./all_somclinical_masks/*.PNG')
# %%

new_path = '/home/sadee/ogevaert/zhang/all_somclinical_masks_png/'

for mf in tqdm(mask_files):
    shutil.copy(mf,new_path + os.path.splitext(os.path.basename(mf))[0] + '.png')
# %%
