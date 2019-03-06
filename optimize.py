from utils import *
from tqdm import tqdm

def perturbation(img_path ,method,radius):
    img=Image.open(img_path).convert('RGB')


    img_shape=np.shape(img)


    if method=='noise':
        noise=np.random.normal(0, 25.5, img_shape)
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
    def  __init__(self,model,original_img,perturbed_img,factor,iter,lr,tv_coeff,tv_beta,\
                  l1_coeff,save_path):
        self.model=model
        self.original_img=original_img
        self.perturbed_img=perturbed_img
        self.factor=factor
        self.iter=iter
        self.lr=lr
        self.tv_coeff=tv_coeff
        self.tv_beta=tv_beta
        self.l1_coeff=l1_coeff
        self.save_path=save_path

    def upsample(self,img):
        if cuda_available():
            upsample=F.interpolate(img,size=(self.original_img.size(2), \
                                    self.original_img.size(3)),mode='bilinear').cuda()
        else:
            upsample = F.interpolate(img, size=(self.original_img.size(2), \
                                                self.original_img.size(3)),mode='bilinear')
        return upsample
    def build(self):

        mask_init=np.random.rand(self.original_img.size(2)/self.factor,\
                                  self.original_img.size(3)/self.factor)

        mask=numpy_to_torch(mask_init)


        output=self.model(self.original_img)

        class_index=np.argmax(output.data.cpu().numpy())

        optimizer=torch.optim.Adam([mask],self.lr)


        for i in tqdm(range(self.iter)):
            upsampled_mask = self.upsample(mask)


            mask_img=torch.mul(upsampled_mask,self.original_img)+torch.mul((1-upsampled_mask),\
                                                                          self.perturbed_img)

            mask_output=torch.nn.Softmax()(self.model(mask_img))
            mask_prob =mask_output[0,class_index]

            loss=self.l1_coeff*torch.mean(1-torch.abs(mask))+\
                 TV(mask,self.tv_coeff,self.tv_beta)+mask_prob

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mask.data.clamp_(0, 1)

        gen_mask=self.upsample(mask)


        return gen_mask