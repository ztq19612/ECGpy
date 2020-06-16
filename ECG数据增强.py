from specAugment import spec_augment_tensorflow as spat
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
data_dir=r'/home/ldf/ztq/ptbMAS/train'
list_1 = os.listdir(data_dir)
for fname in list_1:
    fpath = os.path.join(data_dir, fname)
    print(fname)
    data = Image.open(fpath)
    data = np.array(data)
    label = int(fname.split("_")[0])
    new_feature = spat.spec_augment(data, time_warping_para=0, time_masking_para=4, frequency_masking_para=4,
                                    time_mask_num=4, frequency_mask_num=4)

    image = Image.fromarray(new_feature)
    image = image.convert("L")

    image.save(r'/home/ldf/ztq/ptbspec/spec/'+str(fname)+'.jpg')

#feature_map = Image.open(fpath)
#data = np.array(feature_map)
#new_feature=spat.spec_augment(data,time_warping_para=10,time_masking_para=3,frequency_masking_para=2,
                              #time_mask_num=3,frequency_mask_num=3)
#plt.imshow(feature_map)
#plt.show()
#plt.imshow(new_feature)
#plt.show()
