import numpy as np
import torch
from PIL import Image
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from weather_recognition.config.config import Common
from weather_recognition.config.config import Test
from weather_recognition.model.model_mobilenetv4 import MobileNetV4
from weather_recognition.utils.utils_data_loader import get_data_loaders
import pandas as pd

from ptflops import get_model_complexity_info
from torch import nn
from thop import profile
import torch.nn.utils.prune as prune


def smooth_label(lable, length, smooth_factor):
    """convert targets to one-hot format, and smooth them.
    Args:
        target: target in form with [label1, label2, label_batchsize]
        length: length of one-hot format(number of classes)
        smooth_factor: smooth factor for label smooth
    Returns:
        smoothed labels in one hot format
    """
    # one_hot = _one_hot(target, length, value=1  - smooth_factor)

    lable += smooth_factor / length
    return lable
def predict():
    '''
    预测函数
    '''
    model = MobileNetV4("MobileNetV4ConvSmall")
    load_path_network = Test.mobilenet_pretrain_model
    state_dict = torch.load(load_path_network)
    param_key = 'params'
    model.load_state_dict(state_dict[param_key])
    model = nn.DataParallel(model, device_ids=[1, 2, 3]).to('cuda:1')
    # model.to(Common.device)
    model.eval()

    input_img = "./predict_images/backlight_50.jpg"
    image = Image.open(input_img).convert("RGB")
    image = np.transpose(image, (2, 1, 0))  #
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to('cuda:1')
    output = model(image).cpu()
    print(Common.labels[torch.argmax(output,dim=1)])
    # print(torch.log(torch.softmax(output, dim=-1)))

if __name__ == '__main__':
    predict()

