# 训练部分
import torch
from torch.utils.tensorboard import SummaryWriter
from weather_recognition.config.config import Common, Train
from weather_recognition.model.model_resnet import model_out34 as weatherModel
from weather_recognition.utils.utils_data_loader import loadDataFromDir
from torch import optim
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR

# os.environ['CUDA_VISIBLE_DEVICES']='1, 3' 
# device_ids = [0, 1]
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 1. 获取模型
model = weatherModel
# model = nn.DataParallel(model,device_ids=[0,1])
# model.to(device)

model.to(Common.device)

# 2. 定义损失函数
# criterion = nn.CrossEntropyLoss()
# criterion = torch.nn.functional.binary_cross_entropy()
# 3. 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = MultiStepLR(optimizer=optimizer, milestones=[40,70,100,150,200], gamma=0.2)
#####1:lr:0.0001 stride:[40,70,100,150,200],gamma = 0.2/0.5
#####2:lr:0.0005 stride:20,70,100,140,170,200

# 4. 创建writer
writer = SummaryWriter(log_dir="/log/sigmoid/transform4_lr1_0.5_mul", flush_secs=500)
name_list = []
result_list = []
val_name_list = []
val_result_list = []


def get_indexes(arr):
    return [i for i, x in enumerate(arr) if x == 1]

def train(epoch):
    '''
    训练函数
    '''
    RELOAD_CHECKPOINT = 0
    RELOAD_CHECKPOINT_PATH = "/model/sigmoid/sigmoid_transform4_lr1_0.73.pth"
    if RELOAD_CHECKPOINT == 1:
        PATH_TO_PTH_CHECKPOINT_G = torch.load(RELOAD_CHECKPOINT_PATH,map_location=lambda storage, loc: storage).state_dict()
        model.load_state_dict(PATH_TO_PTH_CHECKPOINT_G)

    
    # 1. 获取dataLoader
    loader = loadDataFromDir('train')
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
        threshold = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
        output = np.array(output.detach().cpu())
        label = np.array(label.detach().cpu())
        best_threshold = np.zeros(output.shape[1])
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
        y_pred = np.array([[1 if output[i,j]>=threshold[j] else 0 for j in range(label.shape[1])] for i in range(len(label))])
        correctly_predicted = len([i for i in range(len(label)) if (label[i]==y_pred[i]).sum() == 8])
        # 计算正确预测的个数
        # labels = torch.argmax(label, dim=1)
        # outputs = torch.argmax(output, dim=1)
        
        # for i in range(0, len(labels)):
        #     if labels[i] == outputs[i]:
        #         correctNum += 1
        #         batchCorrectNum += 1
        batchAcc = correctly_predicted / data.size(0)
        total_correctly_predicted = total_correctly_predicted + correctly_predicted
        print("Epoch:{}\t TrainBatchAcc:{} Battern :transform4_lr1_0.5".format(epoch, batchAcc))
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
    epochAcc = total_correctly_predicted / len(loader.dataset)  # 正确率
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
    loader = loadDataFromDir('val')
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
            output = model(data)  # 获取模型输出
            loss = torch.nn.functional.binary_cross_entropy(output, label)  # 计算损失
            epochLoss += loss.item()   # 计算损失之和
            # threshold = np.arange(0.1,0.9,0.1)
            threshold = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
            output = np.array(output.detach().cpu())
            label = np.array(label.detach().cpu())
            best_threshold = np.zeros(output.shape[1])
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
            y_pred = np.array([[1 if output[i,j]>=threshold[j] else 0 for j in range(label.shape[1])] for i in range(len(label))])
            correctly_predicted = len([i for i in range(len(label)) if (label[i]==y_pred[i]).sum() == 8])
            # loss = criterion(output, label)  # 计算损失
            # loss = torch.sum(-torch.sum(torch.mul(torch.log(torch.softmax(output, dim=-1)), label), dim=-1),dim=0) / 8.0##手动计算
            # epochLoss += loss.item() * data.size(0)  # 计算损失之和
            # 计算正确预测的个数
            # labels = torch.argmax(label, dim=1)
            # outputs = torch.argmax(output, dim=1)
            # for i in range(0, len(labels)):
            #     if labels[i] == outputs[i]:
            #         correctNum += 1
            #         batchCorrectNum += 1
            # batchAcc = batchCorrectNum / data.size(0)
            batchAcc = correctly_predicted / data.size(0)
            total_correctly_predicted = total_correctly_predicted + correctly_predicted
            print("Epoch:{}\t ValBatchAcc:{}".format(epoch, batchAcc))
            if epoch == 120:
                for i in range(0, len(y_pred)):
                    if  np.array_equal(label[i], y_pred[i]) == 0:
                        name_list.append(name[i])
                        # _, indices = torch.max(output[i], -1)  
                        indexes = get_indexes(y_pred[i])
                        if  len(indexes) != 0:    
                            result_len = len(indexes)
                            for i in range (0, result_len):
                                result = Common.labels[indexes[i]]
                                result_list.append(result)
                        if len(name_list) != len(result_list): 
                            if len(name_list) > len(result_list): 
                                mean_width = "NAN"
                                result_list += (len(name_list)-len(result_list)) * [mean_width] 
                            elif len(name_list) < len(result_list): 
                                mean_length = "NAN" 
                                name_list += (len(result_list)-len(name_list)) * [mean_length]
                        
                        dataframe = pd.DataFrame({'name':name_list,'loss': result_list})
                        dataframe.to_csv("../weather_recognition/data/sigmoid/val_sigmoid_transform4_lr1_0.5_mul_.csv", index=False)

        epochLoss = epochLoss   # 平均损失
        epochAcc = total_correctly_predicted / len(loader.dataset)  # 正确率
        print("Epoch:{}\t Loss:{} \t Acc:{}".format(epoch, epochLoss, epochAcc))
        writer.add_scalar("val_loss", epochLoss, epoch)  # 写入日志
        writer.add_scalar("val_acc", epochAcc, epoch)  # 写入日志

    return epochAcc

if __name__ == '__main__':
    maxAcc = 0
    for epoch in range(1,Train.epochs + 1):
        trainAcc = train(epoch)
        valAcc = val(epoch)
        if valAcc > maxAcc:
            maxAcc = valAcc
            # 保存最大模型
            torch.save(model, Train.modelDir + "sigmoid_" + "transform4_lr1_0.5_mul_"  +str(valAcc) + ".pth")
    # 保存模型
    torch.save(model,Train.modelDir+"sigmoid_"+ "transform4_lr1_0.5_mul_" + str(valAcc) + ".pth")


