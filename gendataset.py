import os
from nii import *
import scipy.ndimage
import numpy as np

def findFile(dirpth,ends = ".nii.gz"):
    filelist = []
    for filename in os.listdir(dirpth):
        if filename.endswith(ends):
            filelist.append(os.path.join(dirpth,filename))
    return filelist

def nomaxshape(image,maxshape):
    s = maxshape
    pad = [max(image.shape[0]-s[0],0),
            max(image.shape[1]-s[1],0),
            max(image.shape[2]-s[2],0)]
    if sum(pad)!=0:
        pad1 = ((pad[0]//2,pad[0]-pad[0]//2),(pad[1]//2,pad[1]-pad[1]//2),(pad[2]//2,pad[2]-pad[2]//2))
        image = image[pad1[0][0]:image.shape[0]-pad1[0][1],pad1[1][0]:image.shape[1]-pad1[1][1],pad1[2][0]:image.shape[2]-pad1[2][1]]
    return image

def re_nomaxshape(image,orishape):
    s = image.shape
    pad = [max(orishape[0]-s[0],0),
            max(orishape[1]-s[1],0),
            max(orishape[2]-s[2],0)]
    if sum(pad)!=0:
        pad1 = ((pad[0]//2,pad[0]-pad[0]//2),(pad[1]//2,pad[1]-pad[1]//2),(pad[2]//2,pad[2]-pad[2]//2))
        image = np.pad(image,pad1,'constant',constant_values=0)
    return image

def findheight(img3d,thres = 0):
    count = 0
    start = 0
    end = 0
    for i in range(len(img3d[:])):
        if np.sum(img3d[i,:,:]!=0)>thres:
            if count == 0:
                start = i
            if i == (start + count):
                count += 1
        else:
            if count > len(img3d[:])*0.01:
                end = i
            else:
                start = 0
                end = 0
                count = 0
    end = start + count
    # print("起始位置：",start,"结束位置：",end)
#     if start is not 0:
#         img3d[0:start-1,:,:] = 0
#     if end is not len(img3d[:]):
#         img3d[end+1:,:,:] = 0
    return start, end

def findrange(data_ori):
    data = data_ori.copy()
    # data,spac,ori = nii2array(path)
    data[np.equal(data,2)] = 1
    data = np.transpose(data, (2, 0, 1))
    start,end = findheight(data)
    # print("上",start,"下：",end)
    data = np.transpose(data, (2, 0, 1))
    front,back = findheight(data)
    # print("前：",front,"后：",back)
    data = np.transpose(data, (2, 0, 1))
    l1,r1 = findheight(data)
    data[0:r1,:,:] = 0
    l2,r2 = findheight(data)
    return (start,end,front,back,l1,r1,l2,r2)

masked_pth = "Training/FLARE22_LabeledCase50/images"
masks_pth = "Training/FLARE22_LabeledCase50/labels"
# unmasked_pth = "Training/FLARE22_UnlabeledCase1-1000"

masked_data = findFile(masked_pth)
masks = findFile(masks_pth)
# unmasked_data = findFile(unmasked_pth)

savepth_masked = "dataset/V2/data"
savepth_masks = "dataset/V2/mask"
savepth_unmasked = "dataset/V2/unmask"
target_spacing = [3.5,1.6,1.6]
no_max_shape = [999,170,250]

if not os.path.exists(savepth_masked):
    os.makedirs(savepth_masked)
if not os.path.exists(savepth_masks):
    os.makedirs(savepth_masks)
if not os.path.exists(savepth_unmasked):
    os.makedirs(savepth_unmasked)


for i,pth in enumerate(masked_data):
    data,spac,origin = nii2array(pth)
    data = scipy.ndimage.interpolation.zoom(data, (np.array((spac[2],spac[1],spac[0]))/np.array(target_spacing)), order=0, mode='constant').astype(np.int16) 
    savename = os.path.join(savepth_masked,os.path.basename(pth))
    spacing_f = [target_spacing[2],target_spacing[1],target_spacing[0]]
    data = nomaxshape(data,no_max_shape)
    savenii(data,spacing_f,origin,savename)
    print("%s:"%(i+1),data.shape)

spcsum = np.zeros(3)
sizesum = np.zeros(3)
sizemax = np.zeros(3)
for i,pth in enumerate(masks):
    data,spac,origin = nii2array(pth)
    
    data = scipy.ndimage.interpolation.zoom(data, (np.array((spac[2],spac[1],spac[0]))/np.array(target_spacing)), order=0, mode='constant').astype(np.int16) 
    savename = os.path.join(savepth_masks,os.path.basename(pth))
    spacing_f = [target_spacing[2],target_spacing[1],target_spacing[0]]
    data = nomaxshape(data,no_max_shape)
    savenii(data,spacing_f,origin,savename)
    r = findrange(data)
    spcsum += np.array(spac)
    sizesum += np.array([r[5]-r[4],r[3]-r[2],r[1]-r[0]])
    sizemax = np.maximum.reduce([sizemax,np.array([r[5]-r[4],r[3]-r[2],r[1]-r[0]])])
    print("%s:"%(i+1),data.shape,r)
spcsum = spcsum/len(pth)
sizesum = sizesum/len(pth)
print(spcsum,sizesum,sizemax)

