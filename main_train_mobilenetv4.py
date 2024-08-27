# 训练部分
import sys
sys.path.append('../')
import torch
from torch.utils.tensorboard import SummaryWriter
from weather_recognition.config.config import Common, Train
# from model import ResNet
from weather_recognition.utils.utils_data_loader import loadDataFromDir
from torch import optim
import pandas as pd
from torch.optim.lr_scheduler import MultiStepLR
from weather_recognition.model.model_mobilenetv4 import MobileNetV4


# os.environ['CUDA_VISIBLE_DEVICES']='1, 3' 
# device_ids = [0, 1]
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###One-Hot处理标签
def _one_hot(labels, classes, value=1):
    """
        Convert labels to one hot vectors
    Args:
        labels: torch tensor in format [label1, label2, label3, ...]
        classes: int, number of classes
        value: label value in one hot vector, default to 1
    Returns:
        return one hot format labels in shape [batchsize, classes]
    """
    
    one_hot = torch.zeros(labels.size(0), classes)

    #labels and value_added  size must match
    labels = labels.view(labels.size(0),  -1).to(torch.int64)
    value_added = torch.Tensor(labels.size(0),  1).fill_(value)
    value_added = value_added
    one_hot = one_hot.to(torch.int64)

    one_hot.scatter_add_(1, labels, value_added)
    return one_hot

 
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

# 1. 获取模型
model = MobileNetV4("MobileNetV4ConvSmall")
# model = ResNet([3, 4, 6, 3])
# model = nn.DataParallel(model,device_ids=[0,1])
# model.to(device)

model.to(Common.device)

# 2. 定义损失函数
# criterion = nn.CrossEntropyLoss()
# criterion = torch.nn.functional.binary_cross_entropy()
# 3. 定义优化器
# optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.00025)


##########Resnet learnning rate
# optimizer = optim.Adam( model.parameters(), lr=0.001)
# scheduler = MultiStepLR(optimizer=optimizer, milestones=[10,30,60,100,150,200], gamma=0.5)
# scheduler = MultiStepLR(optimizer=optimizer, milestones=[10,60,100,150,200], gamma=0.5)
#####1:lr:0.0001 stride:[40,70,100,150,200],gamma = 0.2/0.5
#####2:lr:0.0005 stride:20,70,100,140,170,200
#####3:lr:0.0005 stride:10,30,60,100,150,200
#####4:lr:0.0005 stride:10,80,100,150,200 

########MobileV4 learning rate
optimizer = optim.Adam( model.parameters(), lr=0.001) 
scheduler = MultiStepLR(optimizer=optimizer, milestones=[70,100,150,200], gamma=0.5)
#####1:lr1:0.0001 stride1:[40,70,100,150,200],gamma = 0.5
#####2:lr2:0.0005 strid2e:20,70,100,140,170,200
#####3:lr2:0.0005 stride3:10,30,60,100,150,200
#####4:lr2:0.0005 stride4:10,80,100,150,200
#####5:lr1:0.001:stride5:[70,100,150,200]


# 4. 创建writer
writer = SummaryWriter(log_dir="../MobileNetV4_pytorch/log/data_final_resize_val4", flush_secs=500)
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
    RELOAD_CHECKPOINT_PATH = "../MobileNetV4_pytorch/data_final/mobilev4_val5_0.8129770992366412.pth" 
    if RELOAD_CHECKPOINT == 1:
        PATH_TO_PTH_CHECKPOINT_G = torch.load(RELOAD_CHECKPOINT_PATH,map_location=lambda storage, loc: storage).state_dict()
        model.load_state_dict(PATH_TO_PTH_CHECKPOINT_G)
        
    # for name, param in model.named_parameters():
    #     if 'net' in name:
    #         param.requires_grad = False
    
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
        label = smooth_label(label,8,0.1)
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
        print("Epoch:{}\t TrainBatchAcc:{} Battern :".format(epoch, batchAcc))
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
    epochAcc = correctNum / len(loader.dataset)  # 正确率
    print("Epoch:{}\t Loss:{} \t Acc:{}".format(epoch, epochLoss, epochAcc))
    print("transformer3_lr:0.005")
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
            if epoch == 30:
                for i in range(0,len(labels)):
                    if labels[i] != outputs[i]:
                        print(outputs[i])
                        result = Common.labels[outputs[i].item()]
                        name_list.append(name[i])
                        result_list.append(result)
                dataframe = pd.DataFrame({'name':name_list,'loss': result_list})
                # dataframe.to_csv("../MobileNetV4_pytorch/csv/val4_60.csv", index=False)
            # if epoch == 80:
            #     for i in range(0,len(labels)):
            #         if labels[i] != outputs[i]:
            #             print(outputs[i])
            #             result = Common.labels[outputs[i].item()]
            #             name_list.append(name[i])
            #             result_list.append(result)
            #     dataframe = pd.DataFrame({'name':name_list,'loss': result_list})
                # dataframe.to_csv("../MobileNetV4_pytorch/csv/val2_60.csv", index=False)

                        
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
        epochAcc = correctNum / len(loader.dataset)  # 正确率
        print("Epoch:{}\t Loss:{} \t Acc:{}".format(epoch, epochLoss, epochAcc))
        writer.add_scalar("val_loss", epochLoss, epoch)  # 写入日志
        writer.add_scalar("val_acc", epochAcc, epoch)  # 写入日志

    return epochAcc

if __name__ == '__main__':
    maxAcc = 0.81
    for epoch in range(0,Train.epochs + 1):
        trainAcc = train(epoch)
        valAcc = val(epoch)
        if valAcc > maxAcc:
            maxAcc = valAcc
            # 保存最大模型
            torch.save(model, Train.modelDir + "mobilev4_numpy_val4_" +str(valAcc) + ".pth")
    # 保存模型
    torch.save(model,Train.modelDir+ "mobilev4_numpy_val4_" + str(epoch) + "_"+str(valAcc) + ".pth")



