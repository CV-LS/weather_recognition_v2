import sys
# 自定义数据加载器
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from weather_recognition.config.config import Common
from weather_recognition.config.config import Train
from weather_recognition.config.config import Test
import os
from PIL import Image
import torch.utils.data as Data
import numpy as np
import cv2

# 定义数据处理transform

transform = transforms.Compose([
    transforms.Resize(Common.imageSize),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_N = transforms.Compose([
    transforms.Resize(Common.imageSize),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
#######################色调调整、水平翻转####################
transform_h = transforms.Compose([
    # transforms.Resize(Common.imageSize),
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.ToTensor()
])

transform_1 = transforms.Compose([
    transforms.Resize(Common.imageSize),
    transforms.ColorJitter(brightness=0.5, hue=0.3),
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.ToTensor()
])
#########################随机裁剪以0.25的概率发送。
transform_2 = transforms.Compose([
    transforms.Resize(Common.imageSize),
    transforms.RandomApply(transforms=[transforms.RandomCrop(size=(256, 256))], p=0.25),
    # transforms.RandomRotation(15, center=(0, 0), expand=True),
    transforms.Resize(Common.imageSize),
    transforms.ToTensor()
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
#########################色调调整、水平翻转、随机裁剪0.25和随机旋转
transform_3 = transforms.Compose([
    transforms.Resize(Common.imageSize),
    transforms.ColorJitter(brightness=0.5, hue=0.3),
    transforms.RandomHorizontalFlip(p=0.25),
    # transforms.RandomApply(transforms=[transforms.RandomCrop(size=(256, 256))], p=0.25),
    transforms.RandomRotation(15, center=(0, 0), expand=True),
    transforms.Resize(Common.imageSize),
    transforms.ToTensor()
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
##########################色调调整、水平翻转、随机裁剪0.25和随机旋转、Normalize
transform_3_N = transforms.Compose([
    # transforms.Resize(Common.imageSize),
    transforms.ColorJitter(brightness=0.5, hue=0.3),
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.RandomApply(transforms=[transforms.RandomCrop(size=(256, 256))], p=0.2),
    transforms.RandomRotation(15, center=(0, 0), expand=True),
    transforms.Resize(Common.imageSize),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
#############################水平翻转、随机裁剪0.25
transform_4 = transforms.Compose([

    # transforms.Resize((448,448)),
    transforms.RandomHorizontalFlip(p=0.25),
    # transforms.RandomApply(transforms=[transforms.RandomCrop(size=(256, 256))], p=0.25),
    transforms.Resize(Common.imageSize),
    transforms.ToTensor()
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
#############################色调调整、随机裁剪0.25
transform_5 = transforms.Compose([
    transforms.Resize(Common.imageSize),
    transforms.ColorJitter(brightness=0.5, hue=0.3),
    transforms.RandomApply(transforms=[transforms.RandomCrop(size=(256, 256))], p=0.25),
    transforms.Resize(Common.imageSize),
    transforms.ToTensor()
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def data_narmal(Orignal_data):
    d_min = Orignal_data.min()
    if d_min < 0:
        Orignal_data += torch.abs(d_min)
        d_min = Orignal_data.min()
    d_max = Orignal_data.max()
    dst = d_max - d_min
    norm_data = (Orignal_data - d_min).true_divide(dst)
    return norm_data


def gamma_transform_sv(img, gamma1, gamma2):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    illum = hsv[..., 2] / 255.
    illum = np.power(illum, gamma1)
    v = illum * 255.
    v[v > 255] = 255
    v[v < 0] = 0
    hsv[..., 2] = v.astype(np.uint8)

    illum = hsv[..., 1] / 255.
    illum = np.power(illum, gamma2)
    v = illum * 255.
    v[v > 255] = 255
    v[v < 0] = 0
    hsv[..., 1] = v.astype(np.uint8)

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def loadDataFromDir(split):
    '''
    从文件夹中获取数据
    '''
    images = []
    labels = []
    name = []
    # 1. 获取根文件夹下所有分类文件夹
    if split == "train":
        basePath = Common.Train_basePath
    elif split == "val":
        basePath = Common.Val_basePath
    elif split == "test":
        basePath = Common.Test_basePath

    for d in os.listdir(basePath):
        for imagePath in os.listdir(os.path.join(basePath , d)):  # 2. 获取某一类型下所有的图片名称
            # 3. 读取文件
            image = Image.open(os.path.join(basePath , d ,imagePath)).convert("RGB")
            # image = transform_3(image) #内部的to_tensor会自动调换w,h,c到c,h,w 测试mobilenet,最大0.64
            # image = transform_h(image) #内部的to_tensor会自动调换w,h,c到c,h,w 测试resnet,最大0.79
            image = np.transpose(image, (2, 1, 0))  #
            image = np.array(image, dtype=np.float32)
            # print(image.shape)
            # raise "hs"

            # image = cv2.imread(os.path.join(basePath , d , imagePath))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = np.transpose(image,(2,1,0))
            # image = torch.tensor(image,dtype=torch.float)
            # image = image[:, :, ::-1].copy()

            # image = np.array(image)
            # if (image.ndim == 2):
            #     # print(imagePath)
            #     image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
            #     image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
            #     image = gamma_transform_sv(image,1.2,1.1)
            #     image = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
            #     image = cv2.cvtColor(image,cv2.COLOR_BGR2Lab)
            # # image = Image.fromarray(image)
            # image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            # image = cv2.cvtColor(image,cv2.COLOR_BGR2Lab)

            # image = Image.fromarray(image)

            # 3.1 图像预处理：雪景更加白
            ##3.1.1
            # 3.2 图像预处理：雾景更加灰

            # 3.3 图像预处理：雨景边缘加强（待分析）

            # 4. 添加到图片列表中
            images.append((image))
            print("加载第" + str(len(images)) + "条数据")
            # 5. 构造label
            # type1,type2,_ = imagePath.split("_")
            # if (type1 == type2):

            #     categoryIndex = Common.labels.index(type1)  # 获取分类下标
            #     label = [0] * 8  # 初始化label
            #     label[categoryIndex] = 1  # 根据下标确定目标值
            # else :
            #     label = [0] * 8
            #     categoryIndex1 = Common.labels.index(type1)
            #     categoryIndex2 = Common.labels.index(type2)
            #     label[categoryIndex1] = 1
            #     label[categoryIndex2] = 1
            #####################################
            type1, type2 = imagePath.split("_")
            categoryIndex = Common.labels.index(type1)
            label = [0] * 8  # 初始化label
            label[categoryIndex] = 0.9
            label = torch.tensor(label, dtype=torch.float)  # 转为tensor张量

            # 6. 添加到目标值列表
            labels.append(label)
            # labels.append(label)
            name.append(imagePath)
            # labels.append(categoryIndex)
            # 7. 关闭资源
            # image.close()
    # 返回图片列表和目标值列表
    return images, labels, name


class WeatherDataSet(Dataset):
    '''
    自定义DataSet
    '''

    def __init__(self):
        '''
        初始化DataSet
        :param transform: 自定义转换器
        '''
        split = "train"
        images, labels, name = loadDataFromDir(split)  # 在文件夹中加载图片
        self.images = images
        self.labels = labels
        self.name = name

    def __len__(self):
        '''
        返回数据总长度
        :return:
        '''
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        name = self.name[idx]
        # image = transform_4(image)
        # image = data_narmal(image)
        return image, label, name


class WeatherDataSet_val(Dataset):
    '''
    自定义DataSet
    '''

    def __init__(self):
        '''
        初始化DataSet
        :param transform: 自定义转换器
        '''
        split = "val"
        images, labels, name = loadDataFromDir(split)  # 在文件夹中加载图片
        self.images = images
        self.labels = labels
        self.name = name

    def __len__(self):
        '''
        返回数据总长度
        :return:
        '''
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        name = self.name[idx]
        # image = transform(image)
        # image = data_narmal(image)
        return image, label, name


class WeatherDataSet_test(Dataset):
    '''
    自定义DataSet
    '''

    def __init__(self):
        '''
        初始化DataSet
        :param transform: 自定义转换器
        '''
        split = "test"
        images, labels, name = loadDataFromDir(split)  # 在文件夹中加载图片
        self.images = images
        self.labels = labels
        self.name = name

    def __len__(self):
        '''
        返回数据总长度
        :return:
        '''
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        name = self.name[idx]
        # image = transform(image)
        # image = data_narmal(image)
        return image, label, name


def splitData(dataset):
    '''
    分割数据集
    :param dataset:
    :return:
    '''
    # 求解一下数据的总量
    total_length = len(dataset)

    # 确认一下将80%的数据作为训练集, 剩下的20%的数据作为测试集
    # train_length = int(total_length * 0.8)
    # validation_length = total_length - train_length

    # 利用Data.random_split()直接切分数据集, 按照80%, 20%的比例进行切分
    dataset1 = Data.random_split(dataset=dataset, lengths=[total_length])
    return dataset1


def get_data_loaders(data_type: str):
    data_type = data_type.lower()
    if data_type == 'train':
        dataset_class = WeatherDataSet
    elif data_type == 'val':
        dataset_class = WeatherDataSet_val
    elif data_type == 'test':
        dataset_class = WeatherDataSet_test
    else:
        raise ValueError("data_type must be 'train', 'val', or 'test'")

    # 创建数据集实例
    dataset = dataset_class()
    batch_size = Train.batch_size if data_type == 'train' or data_type == 'val' else Test.batch_size

    # 使用配置类属性创建 DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size ,
        shuffle=True if data_type == 'train' else False,
        num_workers=Train.num_workers
    )

    return data_loader

