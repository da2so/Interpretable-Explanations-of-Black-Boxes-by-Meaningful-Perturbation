from PIL import Image,ImageFilter
import numpy as np

import torch
from torch.nn import functional as F
from torch.autograd import Variable

from utils import load_model, cuda_available, save_img ,image_preprocessing

def perturbation(img_path ,method,radius):
    img=Image.open(img_path).convert('RGB')

    if method=='noise':
        noise=np.random.normal(0, 25.5, img_shape=np.shape(img))
        img=np.asarray(img)+noise
        img=Image.fromarray(np.uint8(img))
    elif method=='blur':
        if radius==None:
            radius=10
        img=img.filter(ImageFilter.GaussianBlur(radius))
    elif method=='original':
        pass

    return img

def TV(img,tv_coeff,tv_beta):
    tv_loss = tv_coeff * (
            torch.sum(torch.pow(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]),tv_beta)) +
            torch.sum(torch.pow(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]),tv_beta))
    )
    return tv_loss


class Optimize():
    def  __init__(self,model_path,factor,iter,lr,tv_coeff,tv_beta,\
                  l1_coeff,img_path,perturb):
        self.model_path=model_path
        self.factor=factor
        self.iter=iter
        self.lr=lr
        self.tv_coeff=tv_coeff
        self.tv_beta=tv_beta
        self.l1_coeff=l1_coeff
        self.img_path=img_path
        
        self.model=load_model(self.model_path)
        
        self.original_img=perturbation(self.img_path,'original',None)
        self.original_img_tensor=image_preprocessing(self.original_img)


        self.perturbed_img=perturbation(self.img_path,perturb,5)
        self.perturbed_img_tensor=image_preprocessing(self.perturbed_img)


    def upsample(self,img):
        if cuda_available():
            upsample=F.interpolate(img,size=(int(self.original_img_tensor.size(2)), \
                                    int(self.original_img_tensor.size(3))),mode='bilinear'\
                                   ,align_corners=False).cuda()
        else:
            upsample = F.interpolate(img, size=(int(self.original_img_tensor.size(2)), \
                                    int(self.original_img_tensor.size(3))),mode='bilinear'\
                                     ,align_corners=False)
        return upsample
    
    def build(self):
        
        #mask initialization
        b,c,w,h=self.original_img_tensor.shape
        mask_tensor=torch.rand((b,1,int(w/self.factor),int(h/self.factor)))

        if cuda_available():
            mask_tensor = mask_tensor.cuda()
        mask_tensor=Variable(mask_tensor, requires_grad=True)

        output=self.model(self.original_img_tensor)
        #target class for explanations
        class_index=np.argmax(output.data.cpu().numpy())

        optimizer=torch.optim.Adam([mask_tensor],self.lr)

        for i in range(self.iter+1):
            #upsampling mask to fit the shape of mask to the shape of image
            upsampled_mask = self.upsample(mask_tensor)

            #gjttering
            jitter=torch.randn((b,c,w,h))*0.03
            jitter_org_img_tensor=self.original_img_tensor+jitter.cuda()

            mask_img=torch.mul(upsampled_mask,jitter_org_img_tensor)+torch.mul((1-upsampled_mask),\
                                                                          self.perturbed_img_tensor)

            mask_output=torch.nn.Softmax(dim=1)(self.model(mask_img))
            mask_prob =mask_output[0,class_index]

            loss=self.l1_coeff*torch.mean(1-torch.abs(mask_tensor))+\
                 TV(mask_tensor,self.tv_coeff,self.tv_beta)+mask_prob

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #allow the values of mask to be [0,1]
            mask_tensor.data.clamp_(0, 1)

            if i% 20 == 0:
                print(f'[{i}/{self.iter}] Loss: {loss}  Prob for the target class: {mask_prob}')

        save_img(self.upsample(mask_tensor),self.original_img,self.perturbed_img,self.img_path,self.model_path)
