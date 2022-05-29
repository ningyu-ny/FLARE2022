import numpy as np
import torch
import argparse
from myloss_wce import MyLoss
# from lossFuncs import dice_loss
from torch.utils.data import DataLoader
from torch import autograd, optim
import torch.nn as nn
from torchvision.transforms import transforms
from dataset import *
import matplotlib.pyplot as plt
# from matplotlib import cm
import scipy.misc
import time
from os.path import basename
import os
import pandas as pd
import cv2
import gc
from sklearn.model_selection import KFold
from nii import savenii
from maxregiongrowth import RegionGrowthOptimize as rgo
from rois import eraseROI,eraseROI_onehot
# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.ToTensor(),
])
y_transforms = transforms.Compose([
    transforms.ToTensor(),
])

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path+' 创建成功')
    else:
        print(path+' 目录已存在')


def datestr():
    now = time.localtime()
    return '{}{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday)

def timestr():
    now = time.localtime()
    return '{:02}{:02}'.format(now.tm_hour, now.tm_min)

def adjust_learning_rate(optimizer, decay_rate=.99):
    for param_group in optimizer.param_groups:
        if param_group['lr'] > 1e-7:
            param_group['lr'] *= decay_rate

#训练模型
def train():
    model_path = '/media/ningyu/work/model/unet3d_small/w1d01'
    # model_path ='model/unet3d_small/w1d01'
    batch_size = 2
    data_shape = [80,96,176]
    class_num = 13
    criterion = MyLoss(class_num)
    masked_pth = "dataset/V2/data"
    masks_pth = "dataset/V2/mask"
    unmasked_pth = "dataset/V2/unmask"
    masked_data = findFile(masked_pth)
    masks = findFile(masks_pth)
    unmasked_data = findFile(unmasked_pth)
    masked_data = sorted(masked_data)
    masks = sorted(masks)
    kfold = 5
    for k in range(kfold):
        trainlist_data = masked_data[:int(k * len(masked_data) / kfold)]+masked_data[int((k + 1) * len(masked_data) / kfold):]
        trainlist_mask = masks[:int(k * len(masked_data) / kfold)]+masks[int((k + 1) * len(masked_data) / kfold):]
        testlist_data = masked_data[int(k * len(masked_data) / kfold):int((k + 1) * len(masked_data) / kfold)]
        testlist_mask = masks[int(k * len(masked_data) / kfold):int((k + 1) * len(masked_data) / kfold)]
        train_data = MyDataset(trainlist_data,trainlist_mask,class_num,data_shape,x_transforms,y_transforms)
        train_dataloaders = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        test_data = MyDataset(testlist_data,testlist_mask,class_num,data_shape,x_transforms,y_transforms,mode="test")
        test_dataloaders = DataLoader(test_data, batch_size=2, shuffle=False, num_workers=0)
        train_model(criterion,class_num, train_dataloaders,test_dataloaders, k, kfold,model_path=model_path)


def train_model(criterion,class_num, train_dataloaders,test_dataloaders, k,kfold=5,num_epochs=4000,model_path = ''):
    model_path += '/%s_%s/' %(datestr(),k)
    mkdir(model_path)
    from unet3d_small import unet3d
    model = unet3d(1, class_num+1, batch_norm=True, sample=True).to(device)
    # from unet3d_res import unet3d
    # model = unet3d(class_num=class_num+1).to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    model.apply(weights_init)
    if loadpath is not None:
        checkpoint = torch.load(loadpath)
        # model.load_state_dict(checkpoint)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # 手动设置学习率
        # for param_group in optimizer.param_groups:
        #     param_group['lr']=0.00015


    train_loss_list = []
    train_dice_list = []
    epoch_loss_list = []
    test_loss_list = []
    test_dice_list = []
    total_step = 0
    
    for epoch in range(num_epochs):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        print('Epoch %s/%s(lr:%0.6f)'%(epoch + 1, num_epochs,lr))
        print('-' * 10)
        dt_size = len(train_dataloaders.dataset)
        # dt_size_test = len(dataloaders_test.dataset)
        epoch_loss = 0
        epoch_dice=0
        step = 0
        for x,y in train_dataloaders:
            step += 1
            inputs = x.to(device).float()
            inputs.requires_grad_()
            labels = y.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            print("%d/%d,train_loss:" % (step, (dt_size - 1) // train_dataloaders.batch_size + 1),end="")
            loss,dices = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss_list.append(loss.item())
            # train_dice_list.append(dices)
            epoch_loss += loss.item()
            epoch_dice += dices
            
        test_loss=0
        test_dice=0
        # model.eval()
        step_test=0
        with torch.no_grad():
            for x,y in test_dataloaders:#
                step_test += 1
                inputs = x.to(device).float()
                labels = y.to(device).float()
                optimizer.zero_grad()
                outputs = model(inputs)
                print("%d/%d,test_loss:" % (step_test, (len(test_dataloaders.dataset) - 1) // test_dataloaders.batch_size + 1,),end='')
                loss,dices = criterion(outputs, labels)
                test_loss+=loss.item()
                test_dice += dices
        epoch_dice/=(len(train_dataloaders.dataset)/train_dataloaders.batch_size)
        test_dice/=(len(test_dataloaders.dataset)/test_dataloaders.batch_size)
        model.train()
        epoch_loss_list.append(epoch_loss)
        train_dice_list.append(epoch_dice.tolist())
        test_loss_list.append(test_loss*(kfold-1))
        test_dice_list.append(test_dice.tolist())
        step_loss = pd.DataFrame({'step': range(len(train_loss_list)), 'step_loss': train_loss_list})
        step_loss.to_csv(model_path + '/' + 'step_loss.csv',index=False)
        
        print("epoch %d loss:%0.3f,test_loss:%0.3f" % ((epoch+1), epoch_loss, test_loss*(kfold-1)))
        if epoch % 10 == 9:
            adjust_learning_rate(optimizer)
            data_save = {'net':model.state_dict(),'optimizer':optimizer.state_dict()}
            torch.save(data_save, (model_path + '/%s_epoch_%d.pth' %(timestr(),(epoch+1))))
            train_list = np.array(train_dice_list)
            test_list = np.array(test_dice_list)
            try:
                savemap = {'step': range(len(epoch_loss_list)), 'train_loss': epoch_loss_list, 'test_loss': test_loss_list}
                for i in range(len(train_list[0])):
                    savemap["train_%s"%i] = train_list[:,i]
                    savemap["test_%s"%i] = test_list[:,i]
                step_dice = pd.DataFrame(savemap)
                step_dice.to_csv(model_path + '/' + 'epoch_dice.csv',index=False)
                ifnotsave=0
            except:
                print("xsl save failed.")
    plt.plot(epoch_loss_list,label="train")
    plt.plot(test_loss_list,label="test")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(bbox_to_anchor=(0.8, 0.97), loc=2, borderaxespad=0.)
    plt.savefig(model_path+"/accuracy_loss%s.jpg"%k)
    plt.close()
    train_dice_list = np.array(train_dice_list)
    test_dice_list = np.array(test_dice_list)
    ifnotsave=1
    while ifnotsave:
        try:
            savemap = {'step': range(len(epoch_loss_list)), 'train_loss': epoch_loss_list, 'test_loss': test_loss_list}
            for i in range(len(train_dice_list[0])):
                savemap["train_%s"%i] = train_dice_list[:,i]
                savemap["test_%s"%i] = test_dice_list[:,i]
            step_dice = pd.DataFrame(savemap)
            step_dice.to_csv(model_path + '/' + 'epoch_dice.csv',index=False)
            ifnotsave=0
        except:
            input("保存失败，按任意键重试")

    # text_create(model_path+"/fold_%s_train"%k,epoch_loss_list)
    # text_create(model_path+"/fold_%s_test"%k,test_loss_list)
    # text_create(model_path+"/fold_%s_traindice"%k,train_dice_list)
    # text_create(model_path+"/fold_%s_testdice"%k,test_dice_list)
    del optimizer
    del model
    return

def segonly(test_path,model_path,mod="unet3d_bn"):
    data_shape = [64,160,160]
    class_num = 13
    if mod =="unet3d":
        from unet3d import unet3d
        model = unet3d(1,3,batch_norm=False,sample=False).to(device)
    if mod =="unet3d_bn":
        from unet3d_softmax import unet3d
        model = unet3d(1,class_num+1,batch_norm=True,sample=True).to(device)
    if mod =="unet3d_res":
        from unet3d_res import unet3d
        model = unet3d(class_num=4).to(device)
    if mod =="vnet":
        model = vnet.VNet(elu=False, nll=False, num_out=3).to(device)
    if mod =="wnet_nii":
        from wnet_softmax import wnet as unet3d
        model = unet3d(1,4,batch_norm=False,sample=False).to(device)
    if mod =="wnet_niibn":
        from wnet_softmax import wnet as unet3d
        model = unet3d(1,4,batch_norm=True,sample=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage.cuda(0)))
    input_data = findFile(test_path)
    test_data = MyDataset_test(input_data,data_shape,transform=x_transforms,mask_transform=y_transforms)
    test_dataloaders = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    folderlist = os.listdir(test_path)
    model.eval()
    dt_size = len(test_dataloaders.dataset)
    save_path = test_path
    threshold = 0.5
    lossall=0
    i = 0
    for images,spc,ori,x_path,input_shape0,input_shape1 in test_dataloaders:
        # i+=1
        # if i < 5:
        #     continue
        spc=(spc[0].item(),spc[1].item(),spc[2].item())
        ori=(ori[0].item(),ori[1].item(),ori[2].item())
        input_shape0 = (input_shape0[0].item(),input_shape0[1].item(),input_shape0[2].item())
        input_shape1 = (input_shape1[0].item(),input_shape1[1].item(),input_shape1[2].item())
        with torch.no_grad():
            out_images = []
            for i,x in enumerate(images):
                x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
                y = model(x)
                y = y.view((class_num+1,data_shape[0],data_shape[1],data_shape[2]))
                # y = torch.argmax(y, 0)
                # out_images.append(torch.detach(y).cpu().numpy())
                out_images.append(np.transpose(torch.detach(y).cpu().numpy(),[1,2,3,0]))
                print('Finish:%s/%s'%(i+1,len(images)))
            print("")
            out = eraseROI_onehot(out_images,input_shape1,50)
            out = np.argmax(out,3)
            out = re_nominshape(out,input_shape0)
            spc = [spc[2],spc[1],spc[0]]
            savenii(out,spc,ori,x_path[0][:-7]+'predict.nii.gz')
            rgo('%spredict.nii.gz'%x_path[0][:-7])
            print('Finish:%s'%x_path)
            

if __name__ == '__main__':
    loadpath = None#'model/unet3d_small/2251_epoch_870.pth'
    train()

    # niipath = "dataset/testset/"
    # modelpath = "model/unet3d/1528_epoch_1000.pth"
    # mod="unet3d_bn"
    # segonly(niipath,modelpath,mod)