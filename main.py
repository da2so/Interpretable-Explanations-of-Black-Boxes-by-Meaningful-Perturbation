import argparse


from optimize import Optimize

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Meaningful Perturbation PyTorch')
    parser.add_argument('--img_path', type=str, default='examples/flute.jpg', help='Image path')
    #Pretrained model list:{'AlexNet', 'VGG19', 'ResNet50', 'DenseNet169', 'MobileNet' ,'WideResNet50'}
    parser.add_argument('--model_path', type=str, default='MobileNet', help='Choose a pretrained model or saved model (.pt)')
    parser.add_argument('--perturb', type=str, default='blur', help='Choose a perturbation method (blur, noise)')
    parser.add_argument('--tv_coeff',type=float, default='10e-2',help='Coefficient of TV')
    parser.add_argument('--tv_beta',type=float, default='3',help='TV beta value')
    parser.add_argument('--l1_coeff',type=float,default='10e-3',help='L1 regularization')
    parser.add_argument('--factor',type=int,default=8,help='Factor to upsampling')
    parser.add_argument('--lr',type=float,default=0.1,help='Learning rate')
    parser.add_argument('--iter',type=int,default=300,help='Iteration number')

    args = parser.parse_args()
    mask_opt=Optimize(args.model_path,args.factor,args.iter,args.lr,\
                      args.tv_coeff,args.tv_beta,args.l1_coeff,args.img_path,args.perturb,)
    mask_opt.build()



