# Interpretable Explanations of Black Boxes by Meaningful Perturbation

![fig1](./assets/fig1.png)


![Python version support](https://img.shields.io/badge/python-3.6-blue.svg)
![PyTorch version support](https://img.shields.io/badge/PyTorch-1.7.0-red.svg)

:star: Star us on GitHub — it helps!!


PyTorch implementation for *[Interpretable Explanations of Black Boxes by Meaningful Perturbation](https://arxiv.org/abs/1704.03296)*

## Install

You will need a machine with a GPU and CUDA installed.  
Then, you prepare runtime environment:

   ```shell
   pip install -r requirements.txt
   ```

## Use

This codes are baesed on ImageNet dataset

```shell
python main.py --model_path=vgg19 --img_path=examples/catdog.png
```

Arguments:

- `model_path` - Choose a pretrained model in torchvision.models or saved model (.pt)
	- Examples of available list: ['alexnet', 'vgg19', 'resnet50', 'densenet169', 'mobilenet_v2' ,'wide_resnet50_2', ...]
- `img_path` - Image Path
- `perturb` - Choose a perturbation method (blur, noise)
- `tv_coeff` - Coefficient of TV
- `tv_beta` - TV beta value
- `l1_coeff` - L1 regularization
- `factor` - Factor to upsampling
- `lr` - Learning rate
- `iter` - Iteration

### How to use your customized model

If you want to use customized model that has a type 'OrderedDict', you shoud type a code that loads model object.

Search 'load model' function in utils.py and type a code such as:

```shell
from yourNetwork import yourNetwork())
model=yourNetwork()
```

## Understanding this Paper!

:white_check_mark: Check my blog!!
[Here](https://da2so.github.io/2020-08-11-Meaningful_Perturbation/)