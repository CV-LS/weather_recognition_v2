import numpy as np
import torch
from PIL import Image
from weather_recognition.config.config import  Common
from weather_recognition.config.config import  Test
from weather_recognition.model.model_resnet import model_out34
from torch import nn
from thop import profile
import torch.nn.utils.prune as prune
def sparsity(model):
    # Return global model sparsity
    # a用来统计使用的神经元的个数, 也就是参数量个数
    # b用来统计没有使用到的神经元个数, 也就是参数为0的个数
    a, b = 0., 0.
    for p in model.parameters():
        a += p.numel()        # numel()返回数组A中元素的数量
        b += (p == 0).sum()   # 参数为0 表示没有使用到这个神经元参数
    # b / a 即可以反应模型的稀疏程度
    return b / a


def prune_P(model, amount=0.3):
    # Prune model to requested global sparsity
    
    print('Pruning model... ', end='')
    # 对模型中的nn.Conv2d参数进行修剪
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            # 这里会对模块原来的weight构建两个缓存去, 一个是weight_orig(原参数), 另外一个是weight_mask(原参数的掩码)
            # weight_mask掩码参数有0/1构成, 1表示当前神经元不修剪, 0表示修剪当前神经元
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            # 将name+'_orig'与name+'_mask'从参数列表中删除, 也就是将掩码mask作用于原参数上
            # 使name保持永久修剪, 同时去除参数的前向传播钩子(就是不需要前向传播)
            prune.remove(m, 'weight')  # make permanent

    # 测试模型的稀疏性
    print(' %.3g global sparsity' % sparsity(model))

# 功能: 测试模型参数
def model_parms(model):

    input = torch.randn(1,3, 448, 256)
    flops, params = profile(model, inputs=(input,))
    
    print('flops:{}G'.format(flops / 1e9))
    print('params:{}M'.format(params / 1e6))


# 功能: 测试模型剪枝
def model_prune(model):

    # RELOAD_CHECKPOINT_PATH = "../CNN_Pruning/out/JZ_out34_lr1_dataall_0.8326639892904953.pth"
    # # model = load_model()
    # model = torch.load(RELOAD_CHECKPOINT_PATH,map_location=torch.device('cpu'))
    # model_parms(model)

    # model_info(model, verbose=True)
    # print(model)
    result = sparsity(model)
    print("prune before:{}".format(result))

    #剪枝
    prune_P(model)
    result = sparsity(model)
    print("prune after:{}".format(result))
    model_parms(model)
    # print(model)

def predict():
    '''
    预测函数
    '''
    model = model_out34
    load_path_network = Test.resnet_pretrain_model
    state_dict = torch.load(load_path_network)
    param_key = 'params'
    model.load_state_dict(state_dict[param_key])
    model = nn.DataParallel(model,device_ids=[1,2,3]).to('cuda:1')
    # model.to(Common.device)
    model.eval()

    input_img = "./predict_images/backlight_50.jpg"
    image = Image.open(input_img).convert("RGB")
    # image = transform_3(image) #内部的to_tensor会自动调换w,h,c到c,h,w 测试mobilenet,最大0.64
    # image = transform_h(image) #内部的to_tensor会自动调换w,h,c到c,h,w 测试resnet,最大0.79
    image = np.transpose(image, (2, 1, 0))  #
    image = torch.tensor(image,dtype=torch.float32).unsqueeze(0).to('cuda:1')
    output = model(image).cpu()
    print(Common.labels[torch.argmax(output,dim=1)])


if __name__ == '__main__':
    predict()
