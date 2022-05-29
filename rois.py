import SimpleITK as sitk
import numpy as np
from nii import savenii,nii2array

# mask,spacing,origin = nii2array("train/%s/input.nii.gz"%(29))

def cropROI(volume,outputsize,step):
    ROIs = []
    num_x = (volume.shape[0]-outputsize[0]-1)//step+1
    num_y = (volume.shape[1]-outputsize[1]-1)//step+1
    num_z = (volume.shape[2]-outputsize[2]-1)//step+1
    print(num_x,num_y,num_z)
    for i in range(num_z):
        for j in range(num_y):
            for k in range(num_x):
                ROIs.append(volume[k*step:k*step+outputsize[0],
                            j*step:j*step+outputsize[1],
                            i*step:i*step+outputsize[2]])
            ROIs.append(volume[volume.shape[0]-outputsize[0]:,
                        j*step:j*step+outputsize[1],
                        i*step:i*step+outputsize[2]])
        for k in range(num_x):
            ROIs.append(volume[k*step:k*step+outputsize[0],
                                volume.shape[1]-outputsize[1]:,
                                i*step:i*step+outputsize[2]])
        ROIs.append(volume[volume.shape[0]-outputsize[0]:,
                            volume.shape[1]-outputsize[1]:,
                            i*step:i*step+outputsize[2]])
    print(len(ROIs))
    for j in range(num_y):
        for k in range(num_x):
            ROIs.append(volume[k*step:k*step+outputsize[0],
                                j*step:j*step+outputsize[1],
                                volume.shape[2]-outputsize[2]:])
        ROIs.append(volume[volume.shape[0]-outputsize[0]:,
                            j*step:j*step+outputsize[1],
                            volume.shape[2]-outputsize[2]:])
    for k in range(num_x):
        ROIs.append(volume[k*step:k*step+outputsize[0],
                    volume.shape[1]-outputsize[1]:,
                    volume.shape[2]-outputsize[2]:])
    ROIs.append(volume[volume.shape[0]-outputsize[0]:,
                        volume.shape[1]-outputsize[1]:,
                        volume.shape[2]-outputsize[2]:])
    return ROIs
#将截取的ROI恢复回去

def eraseROI(ROIs,inputsize,step):
    volume = np.zeros(inputsize)
    outputsize = ROIs[0].shape
    num_x = (inputsize[0]-outputsize[0]-1)//step+1
    num_y = (inputsize[1]-outputsize[1]-1)//step+1
    num_z = (inputsize[2]-outputsize[2]-1)//step+1
    for i in range(num_z):
        for j in range(num_y):
            for k in range(num_x):
                volume[k*step:k*step+outputsize[0],
                            j*step:j*step+outputsize[1],
                            i*step:i*step+outputsize[2]] = np.maximum.reduce([volume[k*step:k*step+outputsize[0],
                            j*step:j*step+outputsize[1],
                            i*step:i*step+outputsize[2]],ROIs.pop(0)])
            volume[volume.shape[0]-outputsize[0]:,
                        j*step:j*step+outputsize[1],
                        i*step:i*step+outputsize[2]] = np.maximum.reduce([volume[volume.shape[0]-outputsize[0]:,
                        j*step:j*step+outputsize[1],
                        i*step:i*step+outputsize[2]],ROIs.pop(0)])
        for k in range(num_x):
            volume[k*step:k*step+outputsize[0],
                                volume.shape[1]-outputsize[1]:,
                                i*step:i*step+outputsize[2]] = np.maximum.reduce([volume[k*step:k*step+outputsize[0],
                                volume.shape[1]-outputsize[1]:,
                                i*step:i*step+outputsize[2]],ROIs.pop(0)])
        volume[volume.shape[0]-outputsize[0]:,
                            volume.shape[1]-outputsize[1]:,
                            i*step:i*step+outputsize[2]] = np.maximum.reduce([volume[volume.shape[0]-outputsize[0]:,
                            volume.shape[1]-outputsize[1]:,
                            i*step:i*step+outputsize[2]],ROIs.pop(0)])
    print(len(ROIs))
    for j in range(num_y):
        for k in range(num_x):
            volume[k*step:k*step+outputsize[0],
                                j*step:j*step+outputsize[1],
                                volume.shape[2]-outputsize[2]:] = np.maximum.reduce([volume[k*step:k*step+outputsize[0],
                                j*step:j*step+outputsize[1],
                                volume.shape[2]-outputsize[2]:],ROIs.pop(0)])
        volume[volume.shape[0]-outputsize[0]:,
                            j*step:j*step+outputsize[1],
                            volume.shape[2]-outputsize[2]:] = np.maximum.reduce([volume[volume.shape[0]-outputsize[0]:,
                            j*step:j*step+outputsize[1],
                            volume.shape[2]-outputsize[2]:],ROIs.pop(0)])
    for k in range(num_x):
        volume[k*step:k*step+outputsize[0],
                    volume.shape[1]-outputsize[1]:,
                    volume.shape[2]-outputsize[2]:] = np.maximum.reduce([volume[k*step:k*step+outputsize[0],
                    volume.shape[1]-outputsize[1]:,
                    volume.shape[2]-outputsize[2]:],ROIs.pop(0)])
    volume[volume.shape[0]-outputsize[0]:,
                        volume.shape[1]-outputsize[1]:,
                        volume.shape[2]-outputsize[2]:] = np.maximum.reduce([volume[volume.shape[0]-outputsize[0]:,
                        volume.shape[1]-outputsize[1]:,
                        volume.shape[2]-outputsize[2]:],ROIs.pop(0)])
    return volume

def calguassmap(size):
    a = np.zeros(size[0])
    for i in range(size[0]):
        a[i] = (size[0]-1-i)*i*4/((size[0]-1)**2)
    b = np.zeros(size[1])
    for i in range(size[1]):
        b[i] = (size[1]-1-i)*i*4/((size[1]-1)**2)
    c = np.zeros(size[2])
    for i in range(size[2]):
        c[i] = (size[2]-1-i)*i*4/((size[2]-1)**2)
    d1 = a.reshape([-1,1])*b
    d2 = np.expand_dims(d1,2)*c.reshape([1,1,-1])
    # print(d2)
    # print(d2.shape)
    return d2

def eraseROI_onehot(ROIs,inputsize,step):
    volume = np.zeros([inputsize[0],inputsize[1],inputsize[2],ROIs[0].shape[3]])
    outputsize = ROIs[0].shape
    gaussmap = calguassmap(ROIs[0].shape)
    gaussmap = gaussmap.reshape([gaussmap.shape[0],gaussmap.shape[1],gaussmap.shape[2],1])
    num_x = (inputsize[0]-outputsize[0]-1)//step+1
    num_y = (inputsize[1]-outputsize[1]-1)//step+1
    num_z = (inputsize[2]-outputsize[2]-1)//step+1
    for i in range(num_z):
        for j in range(num_y):
            for k in range(num_x):
                volume[k*step:k*step+outputsize[0],
                            j*step:j*step+outputsize[1],
                            i*step:i*step+outputsize[2]] = volume[k*step:k*step+outputsize[0],
                            j*step:j*step+outputsize[1],
                            i*step:i*step+outputsize[2]]+ROIs.pop(0)*gaussmap
            volume[volume.shape[0]-outputsize[0]:,
                        j*step:j*step+outputsize[1],
                        i*step:i*step+outputsize[2]] = volume[volume.shape[0]-outputsize[0]:,
                        j*step:j*step+outputsize[1],
                        i*step:i*step+outputsize[2]]+ROIs.pop(0)*gaussmap
        for k in range(num_x):
            volume[k*step:k*step+outputsize[0],
                                volume.shape[1]-outputsize[1]:,
                                i*step:i*step+outputsize[2]] = volume[k*step:k*step+outputsize[0],
                                volume.shape[1]-outputsize[1]:,
                                i*step:i*step+outputsize[2]]+ROIs.pop(0)*gaussmap
        volume[volume.shape[0]-outputsize[0]:,
                            volume.shape[1]-outputsize[1]:,
                            i*step:i*step+outputsize[2]] = volume[volume.shape[0]-outputsize[0]:,
                            volume.shape[1]-outputsize[1]:,
                            i*step:i*step+outputsize[2]]+ROIs.pop(0)*gaussmap
    print(len(ROIs))
    for j in range(num_y):
        for k in range(num_x):
            volume[k*step:k*step+outputsize[0],
                                j*step:j*step+outputsize[1],
                                volume.shape[2]-outputsize[2]:] = volume[k*step:k*step+outputsize[0],
                                j*step:j*step+outputsize[1],
                                volume.shape[2]-outputsize[2]:]+ROIs.pop(0)*gaussmap
        volume[volume.shape[0]-outputsize[0]:,
                            j*step:j*step+outputsize[1],
                            volume.shape[2]-outputsize[2]:] = volume[volume.shape[0]-outputsize[0]:,
                            j*step:j*step+outputsize[1],
                            volume.shape[2]-outputsize[2]:]+ROIs.pop(0)*gaussmap
    for k in range(num_x):
        volume[k*step:k*step+outputsize[0],
                    volume.shape[1]-outputsize[1]:,
                    volume.shape[2]-outputsize[2]:] = volume[k*step:k*step+outputsize[0],
                    volume.shape[1]-outputsize[1]:,
                    volume.shape[2]-outputsize[2]:]+ROIs.pop(0)*gaussmap
    volume[volume.shape[0]-outputsize[0]:,
                        volume.shape[1]-outputsize[1]:,
                        volume.shape[2]-outputsize[2]:] = volume[volume.shape[0]-outputsize[0]:,
                        volume.shape[1]-outputsize[1]:,
                        volume.shape[2]-outputsize[2]:]+ROIs.pop(0)*gaussmap
    return volume

if __name__ == '__name__':
    mask,spacing,origin = nii2array("train/%s/input.nii.gz"%(29))
    rois = cropROI(mask,[512,512,80],32)
    outmask = eraseROI(rois,mask.shape,32).astype(np.uint16)
    outmask = np.transpose(outmask,(2,1,0))
    savenii(outmask,spacing,origin,"test",std=True)