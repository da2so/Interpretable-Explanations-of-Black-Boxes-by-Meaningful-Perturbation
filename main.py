import argparse
from optimize import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Meaningful Perturbation')
    parser.add_argument('--img_path', type=str, default='examples/tusker.jpg', help='Image path')
    parser.add_argument('--model', type=str, default='vgg19', help='Choose a model')
    parser.add_argument('--tv_coeff',type=float, default='10e-2',help='Coefficient of TV')
    parser.add_argument('--tv_beta',type=float, default='3',help='TV beta value')
    parser.add_argument('--l1_coeff',type=float,default='10e-3',help='L1 regularization')
    parser.add_argument('--factor',type=int,default=8,help='Factor to upsampling')
    parser.add_argument('--lr',type=float,default=0.1,help='Learning rate')
    parser.add_argument('--iter',type=int,default=300,help='iteration')

    args = parser.parse_args()

    model=load_model()

    original_img=perturbation(args.img_path,'original',None)
    original_img_tensor=image_preprocessing(original_img)

    perturbed_img=perturbation(args.img_path,'blur',5)
    perturbed_img_tensor=image_preprocessing(perturbed_img)

    mask_Opt=Optimize(model,original_img_tensor,perturbed_img_tensor,args.factor,args.iter,args.lr,\
                      args.tv_coeff,args.tv_beta,args.l1_coeff,args.img_path)
    gen_mask=mask_Opt.build()

    save(gen_mask,original_img,perturbed_img,args.img_path)


