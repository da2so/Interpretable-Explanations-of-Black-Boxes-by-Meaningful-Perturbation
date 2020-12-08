import os
import sys

import cv2
import numpy as np

import torchvision.models as models
from torchvision import models, transforms, utils
from torch.autograd import Variable
import torch


def load_model(model_path):
    #for saved model (.pt)
    if '.pt' in model_path:
        if torch.typename(torch.load(model_path)) == 'OrderedDict':
            
            #if you want to use customized model that has a type 'OrderedDict',
            #you shoud load model object as follows:
            
            #from Net import Net()
            #model=Net()
            model.load_state_dict(torch.load(model_path))

        else:
            model=torch.load(model_path)

    #for pretrained model (ImageNet)
    elif hasattr(models , model_path):
        model = getattr(models,model_path)(pretrained=True)
    else:
        print('Choose an available model')
        sys.exit()
        

    model.eval()
    if cuda_available():
        model.cuda()

    return model

def cuda_available():
    use_cuda = torch.cuda.is_available()
    return use_cuda


def image_preprocessing(img):

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    if cuda_available():
        img=preprocess(img).cuda()
    else:
        img=preprocess(img)

    img.unsqueeze_(0)

    return Variable(img, requires_grad=False)


def save_img(mask, img, blurred, img_path,model_path):
    img=np.asarray(img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = np.float32(img) / 255


    blurred = np.asarray(blurred)

    blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB)
    blurred = cv2.resize(blurred, (224, 224))
    blurred=np.float32(blurred)/255

    mask = mask.cpu().data.numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))

    mask = (mask - np.min(mask)) / np.max(mask)
    mask = 1 - mask
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

    heatmap = np.float32(heatmap) / 255

    cam = 1.0 * heatmap + img
    cam = cam / np.max(cam)

    # img = np.float32(img) / 255
    perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred)

    index = img_path.find('/')
    index2 = img_path.find('.')
    path = 'result/' + img_path[index + 1:index2] +'/'+model_path

    if not (os.path.isdir(path)):
        os.makedirs(path)

    original_path = path + "/original.png"
    perturbated_path = path + "/perturbated.png"
    heatmap_path = path + "/heat.png"
    mask_path = path + "/mask.png"
    cam_path = path + "/cam.png"
    cv2.imwrite(original_path, np.uint8(img*255))
    cv2.imwrite(perturbated_path, np.uint8(255 * perturbated))
    cv2.imwrite(heatmap_path, np.uint8(255 * heatmap))
    cv2.imwrite(mask_path, np.uint8(255 * mask))
    cv2.imwrite(cam_path, np.uint8(cam*255))