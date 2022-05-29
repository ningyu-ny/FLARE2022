import torch.utils.data as data
import PIL.Image as Image
import os
from glob import glob
import cv2
import numpy as np
import torch
import SimpleITK as sitk
from nii import nii2array
from rois import cropROI
from dataaugment import sitk_aug
import random
import scipy.ndimage

def findFile(dirpth,ends = ".nii.gz"):
    filelist = []
    for filename in os.listdir(dirpth):
        if filename.endswith(ends) and "predict" not in filename:
            filelist.append(os.path.join(dirpth,filename))
    return filelist
    
def make_dataset_train(root):
    datasets = []
    for dirName,subdirList,fileList in os.walk(root):
        data_filelist = []
        # mask_filelist = []
        for filename in fileList:
            if "input.nii.gz" in filename.lower(): #判断文件是否为dicom文件
                filepth = os.path.join(dirName,filename)
                data_filelist.append([[filepth],[filepth[:-12]+'mask.nii.gz']]) # 加入到列表中
            # if "mask.nii.gz" in filename.lower(): #判断文件是否为dicom文件
                # data_filelist.append([os.path.join(dirName,filename)]) # 加入到列表中

        if len(data_filelist)<1:
            continue
        datasets.extend(data_filelist)
    return datasets

def merge_images_train(files):
    image_depth = len(files)
    image_sample = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    image_height, image_width = image_sample.shape
    image_3d = np.empty((image_depth,image_height, image_width))
    index = 0
    for file in files:
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        image_3d[index, :, :] = image
        index += 1
    return image_3d


def hu_aug(img):
    mean = 0
    mean = img.mean()
    var = np.random.uniform(0,0.001)#高斯噪声
    noise = np.random.normal(0, var ** 0.5, img.shape)
    factor = np.random.uniform(0.8,1.2)#对比度
    delta = np.random.uniform(-0.2,0.2)#亮度
    gamma = np.random.uniform(1,1.4)**random.sample([-1,1],1)[0]#gamma
    # img = ndimage.median_filter(img,(3,3,3))#中值滤波
    img += (noise*(img.max()-img.min())/2).astype(np.int16)#高斯噪声
    img = ((img - mean) * factor + mean).astype(np.int16)#对比度
    img = (img + delta*(img.max()-img.min())).astype(np.int16)#亮度
    img = np.power((img-img.min())/float(img.max()-img.min()), gamma)*(img.max()-img.min())+img.min()#gamma
    return img.astype(np.int16)

def nominshape(image,minshape):
    s = minshape
    pad = [max(s[0]-image.shape[0],0),
            max(s[1]-image.shape[1],0),
            max(s[2]-image.shape[2],0)]
    if sum(pad)!=0:
        pad1 = ((pad[0]//2,pad[0]-pad[0]//2),(pad[1]//2,pad[1]-pad[1]//2),(pad[2]//2,pad[2]-pad[2]//2))
        image = np.pad(image,pad1,'constant',constant_values=0)
    return image

def re_nominshape(image,orishape):
    s = image.shape
    pad = [max(s[0]-orishape[0],0),
            max(s[1]-orishape[1],0),
            max(s[2]-orishape[2],0)]
    if sum(pad)!=0:
        pad1 = ((pad[0]//2,pad[0]-pad[0]//2),(pad[1]//2,pad[1]-pad[1]//2),(pad[2]//2,pad[2]-pad[2]//2))
        image = image[pad1[0][0]:s[0]-pad1[0][1],pad1[1][0]:s[1]-pad1[1][1],pad1[2][0]:s[2]-pad1[2][1]]
    return image

class MyDataset(data.Dataset):
    def __init__(self, data,mask, classnum,data_shape, transform=None, mask_transform=None,mode="train"):
        self.data = data
        self.mask = mask
        self.transform = transform
        self.mask_transform = mask_transform
        self.classnum = classnum
        self.data_shape = data_shape
        self.mode = mode

    def __getitem__(self, index):
        x_path = self.data[index]
        y_path = self.mask[index]
        image = sitk.ReadImage(x_path)
        mask = sitk.ReadImage(y_path)
        if self.mode=="train":
            image,mask = sitk_aug(image,mask)
        
        #当需原图小于要截取的矩阵尺寸时，进行pad
        image = sitk.GetArrayFromImage(image)
        mask = sitk.GetArrayFromImage(mask)
        s = self.data_shape
        pad = [max(s[0]-image.shape[0],0),
                max(s[1]-image.shape[1],0),
                max(s[2]-image.shape[2],0)]
        if sum(pad)!=0:
            pad1 = ((pad[0]//2,pad[0]-pad[0]//2),(pad[1]//2,pad[1]-pad[1]//2),(pad[2]//2,pad[2]-pad[2]//2))
            image = np.pad(image,pad1,'constant',constant_values=0)
            mask = np.pad(mask,pad1,'constant',constant_values=0)
        idx = [random.randint(0,image.shape[0]-s[0]),
                random.randint(0,image.shape[1]-s[1]),
                random.randint(0,image.shape[2]-s[2])]
        image = image[idx[0]:idx[0]+s[0],idx[1]:idx[1]+s[1],idx[2]:idx[2]+s[2]]
        mask = mask[idx[0]:idx[0]+s[0],idx[1]:idx[1]+s[1],idx[2]:idx[2]+s[2]]
        if self.mode=="train":
            image = hu_aug(image)
        image = torch.tensor((image), dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask.astype(np.uint8), dtype=torch.long)
        classnum = self.classnum
        mask = torch.zeros((int(classnum)+1, mask.shape[0], mask.shape[1], mask.shape[2]), dtype=torch.short).scatter_(0, mask.unsqueeze(0), 1)
        return image, mask

    def __len__(self):
        return len(self.data)

class MyDataset_test(data.Dataset):
    def __init__(self, datasets,data_shape, transform=None, mask_transform=None):
        self.datasets = datasets
        self.data_shape = data_shape
        self.transform = transform
        self.mask_transform = mask_transform

    def __getitem__(self, index):
        # target_spacing = [3.41,1.71,1.71]
        x_path = self.datasets[index]
        array_size = np.array(self.data_shape)
        image,spac,ori = nii2array(x_path)
        orishape = image.shape
        image = nominshape(image,array_size)
        # image = scipy.ndimage.interpolation.zoom(image, (np.array((spac[2],spac[1],spac[0]))/np.array(target_spacing)), order=0, mode='constant').astype(np.int16) 
        # image = torch.tensor(image.astype(np.int16), dtype=torch.float32).unsqueeze(0)
        rois_arrays = cropROI(image,array_size,50)#暂时注释
        # image = torch.tensor(image.astype(np.int32), dtype=torch.float32).unsqueeze(0)#暂时删除
        # if self.transform is not None:
        # image = (image/image.max()*255).astype(np.uint8)
        # image = self.transform(image)
        # height, width, depth = image.shape
        # image = image.reshape(1, width, depth,height)
    # if self.mask_transform is not None:

        return rois_arrays,spac,ori,x_path,orishape,image.shape

    def __len__(self):
        return len(self.datasets)
