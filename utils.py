import torchvision.models as models
import cv2
import  numpy as np
from torchvision import models, transforms, utils
from torch.autograd import Variable
from PIL import Image,ImageFilter
import torch
from torch.nn import functional as F
import os

def load_model():
    model = models.vgg19(pretrained=True)

    if cuda_available():
        model.cuda()

    model.eval()
    return  model


def cuda_available():
    use_cuda = torch.cuda.is_available()
    return use_cuda

def load_image(path):
    img = cv2.imread(path, 1)
    img = np.float32(img) / 255

    return img

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
    """
    img=img.view(224,224,3).data.cpu().numpy()
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    img.unsqueeze_(0)

    return img

def numpy_to_torch(img, requires_grad=True):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if cuda_available():
        output = output.cuda()

    output.unsqueeze_(0)
    v = Variable(output, requires_grad=requires_grad)
    return v


def save(mask, img, blurred, path):
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

    index = path.find('/')
    index2 = path.find('.')
    path = 'result/' + path[index + 1:index2]

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