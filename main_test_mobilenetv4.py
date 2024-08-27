import torch
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



# 功能: 测试模型参数
def model_parms(model):
    input = torch.randn(1, 3, 448, 256)
    flops, params = profile(model, inputs=(input,))

    print('flops:{}G'.format(flops / 1e9))
    print('params:{}M'.format(params / 1e6))

def test():
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


    name_list = []
    result_list = []
    correctNum = 0
    loader = get_data_loaders('test')

    for data, label, name in loader:
        batchCorrectNum = 0
        data, label = data.to('cuda:1'), label.to('cuda:1')
        output = model(data).to('cuda:1')
        labels = torch.argmax(label, dim=1)
        outputs = torch.argmax(output, dim=1)
        for i in range(0, len(labels)):
            if labels[i] == outputs[i]:
                correctNum += 1
                batchCorrectNum += 1
        batchAcc = batchCorrectNum / data.size(0)
        print("TestBatchAcc:{}".format(batchAcc))
        for i in range(0, len(labels)):
            if labels[i] == outputs[i]:
                result = Common.labels[outputs[i].item()]
                name_list.append(name[i])
                result_list.append(result)
        dataframe = pd.DataFrame({'name': name_list, 'result': result_list})
        dataframe.to_csv("results/1.csv", index=False)

    epochAcc = correctNum / len(loader.dataset)  # 正确率
    print('预测的准确率为：{}'.format(epochAcc))


if __name__ == '__main__':
    test()
    """
    预测的准确率为：0.9039207238259371
    """
    model_parms(MobileNetV4("MobileNetV4ConvSmall"))
    """
    flops:0.427003392G
    params:1.211816M
    """
