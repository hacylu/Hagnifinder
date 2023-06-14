import torch
import sys
from PIL import Image
from Regression_Model import ResNet50

import torchvision
import argparse

##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default='../img/breast.png')
args = parser.parse_args()




normalize = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

preprocess = torchvision.transforms.Compose([
    #torchvision.transforms.RandomCrop(224),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize
])


def load_model(name='old'):  # 加载保存的数据集

    if type(name) != str:
        sys.exit('命令必须为字符串！')
    if name == 'new':
        model = ResNet50()  # 自定义网络

        # model = ResNet50()
    elif name == 'old':
        model = torch.load('..\\model_save\\bestmodel.pkl')  # 在此处更改模型参数
    else:
        sys.exit('命令必须是new或者old！')
    return model


def test(model, imgpath):
    model.eval()
    img = Image.open(imgpath)
    image = preprocess(img)
    images1 = image.view(1, *image.size())
    images2 = images1.to(device)
    # text
    out = model(images2)

    return out




if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
    model = load_model('old')
    # model = ResNet50()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    img_path = args.image_path

    outputs = test(model, img_path)
    print('magnification: %.3f x' % outputs)
