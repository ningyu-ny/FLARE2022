import os
from nii import *
import scipy.ndimage
import numpy as np
# l = 5
# a = np.zeros(l)
# for i in range(l):
#     a[i] = (l-1-i)*i*4/((l-1)**2)
# b = a.reshape([-1,1])*a
# c = np.expand_dims(b,1)*a.reshape([-1,1])
# print(c)

# def calguassmap(size):
#     a = np.zeros(size[0])
#     for i in range(size[0]):
#         a[i] = (size[0]-1-i)*i*4/((size[0]-1)**2)
#     b = np.zeros(size[1])
#     for i in range(size[1]):
#         b[i] = (size[1]-1-i)*i*4/((size[1]-1)**2)
#     c = np.zeros(size[2])
#     for i in range(size[2]):
#         c[i] = (size[2]-1-i)*i*4/((size[2]-1)**2)
#     d1 = a.reshape([-1,1])*b
#     d2 = np.expand_dims(d1,2)*c.reshape([1,1,-1])
#     # print(d2)
#     # print(d2.shape)
#     return d2

# result = calguassmap([5,6,7])
def findFile(dirpth,ends = ".nii.gz"):
    filelist = []
    for filename in os.listdir(dirpth):
        if filename.endswith(ends):
            filelist.append(os.path.join(dirpth,filename))
    return filelist

masked_pth = "Training/FLARE22_LabeledCase50/images"
masks_pth = "Training/FLARE22_LabeledCase50/labels"
savepth_masked = "Training/FLARE22_LabeledCase50_fixed/images"
savepth_masks = "Training/FLARE22_LabeledCase50_fixed/labels"

if not os.path.exists(savepth_masked):
    os.makedirs(savepth_masked)
if not os.path.exists(savepth_masks):
    os.makedirs(savepth_masks)

masked_data = findFile(masked_pth)
masks = findFile(masks_pth)

# for i,pth in enumerate(masked_data):
#     data,spac,origin = nii2array(pth)
#     savename = os.path.join(savepth_masked,os.path.basename(pth))
#     savenii(data,spac,origin,savename)
#     print("%s:"%(i+1),data.shape)

for i,pth in enumerate(masks):
    data,spac,origin = nii2array(pth)
    savename = os.path.join(savepth_masks,os.path.basename(pth))
    savenii(data,spac,origin,savename)
    print("%s:"%(i+1),data.shape)