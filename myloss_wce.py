import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MyLoss(nn.Module):
    def __init__(self,num_class):
        super(MyLoss, self).__init__()
        self.num_class = num_class

    def forward(self, inputs, target):
        smooth = 1.
        q = inputs
        p = target
        min = torch.tensor([0.00001]).to(device)
        ce = -p * torch.log(q.max(min))
        dice_mean=(2 * (p*q).sum(dim=(0,2,3,4)) + smooth) / (q.sum(dim=(0,2,3,4)) + p.sum(dim=(0,2,3,4)) + smooth+1e-8)
        diceloss = (1 - dice_mean[1:]).mean()

        myloss = 0.8*torch.mean(ce) + 0.2*diceloss
        dices=torch.detach(dice_mean).cpu().numpy()
        print("%0.6f  ("%myloss.item(),"CE:%04f "%torch.mean(ce).item(),"Dice:%04f )"%dice_mean[1:].mean().item())
        print("Dices:",["%s:%0.3f"%(i,dice) for i,dice in enumerate(dices)])
        
        return myloss,dices

