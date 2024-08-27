# 训练部分
import os
import sys
sys.path.append('../../')
import torch
from torch.utils.tensorboard import SummaryWriter
from weather_recognition.options.config import Common, Train
from weather_recognition.model.model import model_out34 as WeatherModel_out34
# from model import ResNet
from weather_recognition.utils.utils_data_loader import  valLoader
from torch import optim
import pandas as pd
from torch.optim.lr_scheduler import MultiStepLR
from torch.quantization import QuantStub, DeQuantStub

# os.environ['CUDA_VISIBLE_DEVICES']='1, 3'
# device_ids = [0, 1]
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1. 获取模型
model = WeatherModel_out34
print(model)
# model = ResNet([3, 4, 6, 3])
# model = nn.DataParallel(model,device_ids=[0,1])
# model.to(device)

model.to(Common.device)

# 2. 定义损失函数
# criterion = nn.CrossEntropyLoss()
# criterion = torch.nn.functional.binary_cross_entropy()
# 3. 定义优化器
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00025)
optimizer = optim.Adam( model.parameters(), lr=0.001)
# scheduler = MultiStepLR(optimizer=optimizer, milestones=[10,30,60,100,150,200], gamma=0.5)
scheduler = MultiStepLR(optimizer=optimizer, milestones=[10,80,100,150,200], gamma=0.5)
RELOAD_CHECKPOINT = 1
RELOAD_CHECKPOINT_PATH = "../CNN_Pruning/after/JZ_out34_lr1_dataall_smooth_0.8460508701472557.pth" 
if RELOAD_CHECKPOINT == 1:
    model = torch.load(RELOAD_CHECKPOINT_PATH,map_location=lambda storage, loc: storage)
    model.to(Common.device)
#####1:lr:0.0001 stride:[40,70,100,150,200],gamma = 0.2/0.5
#####2:lr:0.0005 stride:20,70,100,140,170,200

# 4. 创建writer
writer = SummaryWriter(log_dir="../weather_recognition/log/JZ/JZ_gamma0.5_calcu", flush_secs=500)
name_list = []
result_list = []
val_name_list = []
val_result_list = []



def quantize_weights(model, bits=8):
    for name, param in model.named_parameters():
        if param.dim() >= 2:  # 仅对权重进行量化，排除偏置项等
            mask = torch.abs(param) > 0  # 创建一个布尔掩码，标记非零参数位置
            nonzero_values = param[mask]  # 获取非零参数值
            min_value = torch.min(nonzero_values)
            max_value = torch.max(nonzero_values)
            qmin = 0.0
            qmax = 2.0 ** bits - 1.0
            scale = (max_value - min_value) / (qmax - qmin)
            zero_point = qmin - min_value / scale
            param.data = torch.round(param / scale + zero_point).clamp(qmin, qmax) * scale  # 量化参数
            param.data[mask.logical_not()] = 0  # 将原始参数为零的位置保持不变
            # param.data = param.data.to(torch.int8)

def replace_forward(module):
    module.quant = QuantStub()
    module.dequant = DeQuantStub()
    raw_forward = module.forward

    def forward(x):
        x = module.quant(x)
        x = raw_forward(x)
        x = module.dequant(x)
        return x
    module.forward = forward

def get_indexes(arr):
    return [i for i, x in enumerate(arr) if x == 1]

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')

def train(epoch):
    '''
    训练函数
    '''
        
    # for name, param in model.named_parameters():
    #     if 'net' in name:
    #         param.requires_grad = False
    
    # 1. 获取dataLoader
    loader = trainLoader
    # 2. 调整为训练状态
    model.train()
    print()
    print('========== Train Epoch:{} Start =========='.format(epoch))
    epochLoss = 0  # 每个epoch的损失
    epochAcc = 0  # 每个epoch的准确率
    correctNum = 0  # 正确预测的数量
    acc = []
    accuracies = []
    total_correctly_predicted = 0
    for data, label,name in loader:
        # data, label= data.to(device), label.to(device) # 加载到对应设备
        # data, label= data.to(device_ids=[0,1]), label.to(device_ids=[0,1]) # 加载到对应设备
        data, label= data.to(Common.device), label.to(Common.device) # 加载到对应设备
        batchAcc = 0  # 单批次正确率
        batchCorrectNum = 0  # 单批次正确个数
        optimizer.zero_grad()  # 清空梯度
        output = model(data)  # 获取模型输出
        # label = label.long()
        loss = torch.nn.functional.binary_cross_entropy(output, label)  # 计算损失
        
        # loss = torch.sum(-torch.sum(torch.mul(torch.log(torch.softmax(output, dim=-1)), label), dim=-1),dim=0) / 8.0##手动计算
        loss.backward()  # 反向传播梯度
        optimizer.step()  # 更新参数
        scheduler.step()
        epochLoss += loss.item()  # 计算损失之和
        # threshold = np.arange(0.1,0.9,0.1)
        # threshold = [0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7]
        # output = np.array(output.detach().cpu())
        # label = np.array(label.detach().cpu())


        # best_threshold = np.zeros(output.shape[1])
        # y_pred = np.array([[1 if output[i,j]>=threshold[j] else 0 for j in range(label.shape[1])] for i in range(len(label))])
        # correctly_predicted = len([i for i in range(len(label)) if (label[i]==y_pred[i]).sum() == 8])
        # 计算正确预测的个数
        labels = torch.argmax(label, dim=1)
        outputs = torch.argmax(output, dim=1)
        
        for i in range(0, len(labels)):
            if labels[i] == outputs[i]:
                correctNum += 1
                batchCorrectNum += 1
        batchAcc = batchCorrectNum / data.size(0)
        print("Epoch:{}\t TrainBatchAcc:{} Battern :transform4_lr1_JZ_0.001_gamma0.5_calcu".format(epoch, batchAcc))
        if epoch == 60:
                for i in range(0,len(labels)):
                    if labels[i] != outputs[i]:
                        print(outputs[i])
                        result = Common.labels[outputs[i].item()]
                        name_list.append(name[i])
                        result_list.append(result)
                dataframe = pd.DataFrame({'name':name_list,'loss': result_list})
                # dataframe.to_csv("../weather_recognition/data/JZ/train_transform4_lr1_0.001_delete_gamma0.5_final_d_out3and4.csv", index=False)
        # if epoch == 200:
        #     for i in range(0, len(y_pred)):
        #         if  np.array_equal(label[i], y_pred[i]) == 0:
        #             name_list.append(name[i])
        #             # _, indices = torch.max(output[i], -1)  
        #             indexes = get_indexes(y_pred[i])
        #             if  len(indexes) != 0:    
        #                 result_len = len(indexes)
        #                 for i in range (0, result_len):
        #                     result = Common.labels[indexes[i]]
        #                     result_list.append(result)
        #             if len(name_list) != len(result_list): 
        #                 if len(name_list) > len(result_list): 
        #                     mean_width = "NAN"
        #                     result_list += (len(name_list)-len(result_list)) * [mean_width] 
        #                 elif len(name_list) < len(result_list): 
        #                     mean_length = "NAN"
        #                     name_list += (len(result_list)-len(name_list)) * [mean_length]
                    
        #             dataframe = pd.DataFrame({'name':name_list,'loss': result_list})
        #             dataframe.to_csv("../weather_recognition/data/sigmoid/train_sigmoid_transform_2lr0.0001.csv", index=False)

    epochLoss = epochLoss  # 平均损失
    epochAcc = correctNum / len(trainLoader.dataset)  # 正确率
    print("Epoch:{}\t Loss:{} \t Acc:{}".format(epoch, epochLoss, epochAcc))
    writer.add_scalar("train_loss", epochLoss, epoch)  # 写入日志
    writer.add_scalar("train_acc", epochAcc, epoch)  # 写入日志
    
    return epochAcc

def val(epoch):
    '''
    验证函数
    :param epoch: 轮次
    :return:
    '''
    # 1. 获取dataLoader
    loader = valLoader
    # 2. 初始化损失、准确率列表
    valLoss = []
    valAcc = []
    # 3. 调整为验证状态
    model.eval()
    print()
    print('========== Val Epoch:{} Start =========='.format(epoch))
    epochLoss = 0  # 每个epoch的损失
    epochAcc = 0  # 每个epoch的准确率
    correctNum = 0  # 正确预测的数量
    acc = []
    accuracies = []
    total_correctly_predicted = 0
    with torch.no_grad():
        name_list = []
        result_list = []
        for data, label ,name in loader:
            data, label= data.to(Common.device), label.to(Common.device)  # 加载到对应设备
            # data, label= data.to(device_ids=[0,1]), label.to(device_ids=[0,1])  # 加载到对应设备
            batchAcc = 0  # 单批次正确率
            batchCorrectNum = 0  # 单批次正确个数
            output = model(data).to(Common.device) # 获取模型输出
            loss = torch.nn.functional.binary_cross_entropy(output, label)  # 计算损失
            epochLoss += loss.item()   # 计算损失之和
            # threshold = np.arange(0.1,0.9,0.1)
            # threshold = [0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7]
            # output = np.array(output.detach().cpu())
            # label = np.array(label.detach().cpu())
            # best_threshold = np.zeros(output.shape[1])
            ##################马修斯计算指标
            # for i in range(output.shape[1]):
            #     y_prob = np.array(output[:,i])
            #     for j in threshold:
            #         y_pred = [1 if prob>=j else 0 for prob in y_prob]
            #         acc.append( matthews_corrcoef(label[:,i],y_pred))
            #     acc   = np.array(acc)
            #     index = np.where(acc==acc.max())
            #     accuracies.append(acc.max())
            #     best_threshold[i] = threshold[index[0][0]]
            #     acc = [] 
            # print("best thresholds",best_threshold)
            # y_pred = np.array([[1 if output[i,j]>=threshold[j] else 0 for j in range(label.shape[1])] for i in range(len(label))])
            # correctly_predicted = len([i for i in range(len(label)) if (label[i]==y_pred[i]).sum() == 8])
            # loss = criterion(output, label)  # 计算损失
            # loss = torch.sum(-torch.sum(torch.mul(torch.log(torch.softmax(output, dim=-1)), label), dim=-1),dim=0) / 8.0##手动计算
            # epochLoss += loss.item() * data.size(0)  # 计算损失之和
            # 计算正确预测的个数
            labels = torch.argmax(label, dim=1)
            outputs = torch.argmax(output, dim=1)
            for i in range(0, len(labels)):
                if labels[i] == outputs[i]:
                    correctNum += 1
                    batchCorrectNum += 1
            batchAcc = batchCorrectNum / data.size(0)
            # batchAcc = correctly_predicted / data.size(0)
            # total_correctly_predicted = total_correctly_predicted + correctly_predicted
            print("Epoch:{}\t ValBatchAcc:{}".format(epoch, batchAcc))
            if epoch == 100:
                for i in range(0,len(labels)):
                    if labels[i] != outputs[i]:
                        print(outputs[i])
                        result = Common.labels[outputs[i].item()]
                        name_list.append(name[i])
                        result_list.append(result)
                dataframe = pd.DataFrame({'name':name_list,'loss': result_list})
                dataframe.to_csv("../weather_recognition/data/JZ/val_transform4_lr1_0.001_delete_gamma0.5_calcu.csv", index=False)

                        
            #     for i in range(0, len(y_pred)):
            #         if  np.array_equal(label[i], y_pred[i]) == 0:
            #             name_list.append(name[i])
            #             # _, indices = torch.max(output[i], -1)  
            #             indexes = get_indexes(y_pred[i])
            #             if  len(indexes) != 0:    
            #                 result_len = len(indexes)
            #                 for i in range (0, result_len):
            #                     result = Common.labels[indexes[i]]
            #                     result_list.append(result)
            #             if len(name_list) != len(result_list): 
            #                 if len(name_list) > len(result_list): 
            #                     mean_width = "NAN"
            #                     result_list += (len(name_list)-len(result_list)) * [mean_width] 
            #                 elif len(name_list) < len(result_list): 
            #                     mean_length = "NAN" 
            #                     name_list += (len(result_list)-len(name_list)) * [mean_length]
                        
                
        epochLoss = epochLoss   # 平均损失
        epochAcc = correctNum / len(valLoader.dataset)  # 正确率
        print("Epoch:{}\t Loss:{} \t Acc:{}".format(epoch, epochLoss, epochAcc))
        writer.add_scalar("val_loss", epochLoss, epoch)  # 写入日志
        writer.add_scalar("val_acc", epochAcc, epoch)  # 写入日志

    return epochAcc

if __name__ == '__main__':
    maxAcc = 0.6
    # for epoch in range(1,Train.epochs + 1):
        # trainAcc = train(epoch)
    valAcc = val(1)
    print("未进行Fusion时模型准确率: \n",valAcc)
    print('the Size (MB) of original resnet18:', os.path.getsize("../CNN_Pruning/after/JZ_out34_lr1_dataall_smooth_0.8460508701472557.pth")/1e6)
    model.eval()
    print('resnet18.layer1: Before fusion \n', model.Resnet_layer1)
###################################################fusion + Quantize########################################################################
    replace_forward(model)
    torch.quantization.fuse_modules(
    model,
    [['conv1', 'bn1'],  
     ['Resnet_layer1.0.conv1', 'Resnet_layer1.0.bn1'],  
     ['Resnet_layer1.0.conv2', 'Resnet_layer1.0.bn2'],
     ['Resnet_layer1.0.conv3', 'Resnet_layer1.0.bn3'],  
     ['Resnet_layer1.0.downsample.0', 'Resnet_layer1.0.downsample.1'],

     ['Resnet_layer1.1.conv1', 'Resnet_layer1.1.bn1'],  
     ['Resnet_layer1.1.conv2', 'Resnet_layer1.1.bn2'],
     ['Resnet_layer1.1.conv3', 'Resnet_layer1.1.bn3'],  

     ['Resnet_layer1.2.conv1', 'Resnet_layer1.2.bn1'],  
     ['Resnet_layer1.2.conv2', 'Resnet_layer1.2.bn2'],
     ['Resnet_layer1.2.conv3', 'Resnet_layer1.2.bn3'],
     
     ['Resnet_layer2.0.conv1', 'Resnet_layer2.0.bn1'],  
     ['Resnet_layer2.0.conv2', 'Resnet_layer2.0.bn2'],
     ['Resnet_layer2.0.conv3', 'Resnet_layer2.0.bn3'],  
     ['Resnet_layer2.0.downsample.0', 'Resnet_layer2.0.downsample.1'],  
     
     ['Resnet_layer2.1.conv1', 'Resnet_layer2.1.bn1'],  
     ['Resnet_layer2.1.conv2', 'Resnet_layer2.1.bn2'],
     ['Resnet_layer2.1.conv3', 'Resnet_layer2.1.bn3'],
     
     ['Resnet_layer2.2.conv1', 'Resnet_layer2.2.bn1'],  
     ['Resnet_layer2.2.conv2', 'Resnet_layer2.2.bn2'],
     ['Resnet_layer2.2.conv3', 'Resnet_layer2.2.bn3'],  
     
     ['Resnet_layer2.3.conv1', 'Resnet_layer2.3.bn1'],  
     ['Resnet_layer2.3.conv2', 'Resnet_layer2.3.bn2'],
     ['Resnet_layer2.3.conv3', 'Resnet_layer2.3.bn3'],  
    ], inplace=True
)
    print('resnet18.layer1: After fusion \n', model.layer1)
    valAcc_fusion = val(1)
    print("进行Fusion后模型准确率: \n",valAcc_fusion)
##############################################################################################################################
    model.qconfig = torch.quantization.default_qconfig
    print("resnet18_model_fusion.qconfig: \n", model.qconfig)
    torch.quantization.prepare(model, inplace=True)
    
    # Calibrate first
    print('Post Training Quantization Prepare: Inserting Observers')
    print('\n resnet18.layer1: After observer insertion \n', model.Resnet_layer1)
    
    valAcc_quantize = val(1)
    print("量化后模型准确度：\n",valAcc_quantize)
    
    model = model.cpu()
    torch.quantization.convert(model, inplace=True)
    print('Post Training Quantization: Convert done')
    print('\n resnet18.layer1: After fusion and quantization, note fused modules: \n', model)
    
    print("Size of model after quantization")
    print_size_of_model(model) 
    
    torch.save(model.state_dict(), Train.Quantizedir + "Quantize_" +str(valAcc_quantize) + ".pth")
    
    model.load_state_dict(
    torch.load(Train.Quantizedir + "Quantize_" +str(valAcc_quantize) + ".pth", map_location="cpu"))
    model = model.state_dict()
    valAcc_quantize = val(1)
    print("量化后模型准确度：\n",valAcc_quantize)
    # torch.save(model, Train.Quantizedir + "Quantize_" + "8bit_"  +str(valAcc_quantize) + ".pth")
    
    # if valAcc > maxAcc:
    #     maxAcc = valAcc
    #     # 保存最大模型
    #     torch.save(model, Train.modelDir + "JZ_" + "transform4_lr1_0.001_deletee_gamma0.5_calcu_"  +str(valAcc) + ".pth")
    # 保存模型
    # torch.save(model,Train.modelDir+ "JZ_"+ "transform4_lr1_0.001_delete_gamma0.5_calcu_" + str(valAcc) + ".pth")


