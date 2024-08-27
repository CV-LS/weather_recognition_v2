import time

import torch
import torch.distributed as dist
# 项目配置文件

class Common:
    '''
    通用配置
    '''
    Train_basePath = "../weather_recognition/trainsets"  # 图片文件基本路径
    Val_basePath = "../weather_recognition/testsets"  # 图片文件基本路径
    Test_basePath = "../weather_recognition/trainsets"  # 图片文件基本路径
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 主要用于设置主设备
    imageSize = (448,256) # 图片大小
    labels = ["backlight","cloudy","fog","indoor","rain","snow","sunny","monochrome"] # 标签名称/文件夹名称

class Train:
    '''
    训练相关配置
    '''
    num_workers = 3  # 对于Windows用户，这里应设置为0，否则会出现多线程错误
    batch_size = num_workers*16
    lr = 0.001
    start_epoch = 1
    epochs = 100
    scheduler_name = "MultiStepLR"
    milestones = [100, 150, 200, 240, 250, 260]
    gamma = 0.5

    logDir = "../weather_recognition/log/" + time.strftime('%Y-%m-%d-%H-%M-%S',time.gmtime()) # 日志存放位置
    modelDir = "../MobileNetV4_pytorch/data_final/" # 模型存放位置
    JZmodeldir = "../MobileNetV4_pytorch/mobilenetv4/"
    Quantizedir = "../CNN_Pruning/Quantizedir/"

class Test:
    '''
    训练相关配置
    '''
    batch_size = 3*32
    num_workers = 3  # 对于Windows用户，这里应设置为0，否则会出现多线程错误
    log = "../weather_recognition/log/" + time.strftime('%Y-%m-%d-%H-%M-%S',time.gmtime()) # 日志存放位置
    mobilenet_pretrain_model = "../weather_recognition/model_zoo/mobilenetv4/mobilenetv4_transform_val_0.9116760017233951_b_192_e_7_lr_3.125e-05.pth" # 模型存放位置
    resnet_pretrain_model = "../weather_recognition/model_zoo/resnet/Resnet_transform_val_0.9004739336492891_b_48_e_137_lr_1.5625e-05.pth" # 模型存放位置



