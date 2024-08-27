# 训练部分
import sys
sys.path.append('../../')
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from weather_recognition.options.config import Common, Train
from weather_recognition.model.model import model_out34 as weatherModel_out34
# from model import ResNet
from weather_recognition.utils.utils_data_loader import trainLoader, valLoader
from torch import optim
import pandas as pd
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable


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

def get_model2(model, learning_rate=1e-3, weight_decay=1e-4):

    # set the first layer not trainable
    # model.features.conv0.weight.requires_grad = False

    # all fc layers
    weights = [
        p for n, p in model.named_parameters()
        if 'weight' in n and 'conv' not in n
    ]

    # all conv layers
    weights_to_be_quantized = [
        p for n, p in model.named_parameters()
        # if 'conv' in n and 'conv0' not in n
        if 'conv' in n and 'weight' in n
    ]

    biases = [
        p for n, p in model.named_parameters()
        if 'bias' in n
    ]    

    params = [
        {'params': weights, 'weight_decay': weight_decay},
        {'params': weights_to_be_quantized, 'weight_decay': weight_decay},
        {'params': biases,  'weight_decay': weight_decay}
    ]
    optimizer = optim.Adam(params, lr=learning_rate, momentum=0.9)
    

    loss = nn.CrossEntropyLoss().to(Common.device)
    model = model.to(Common.device)  # move the model to gpu
    return model, loss, optimizer

def quantize_bw(kernel):
    """
    binary quantization
    Return quantized weights of a layer.
    """
    delta = kernel.abs().mean()
    sign = kernel.sign().float()



    return sign*delta

# 1. 获取模型
model = weatherModel_out34
# model = ResNet([3, 4, 6, 3])
# model = nn.DataParallel(model,device_ids=[0,1])
# model.to(device)

model.to(Common.device)

# 2. 定义损失函数
# criterion = nn.CrossEntropyLoss()
# criterion = torch.nn.functional.binary_cross_entropy()
# 3. 定义优化器
# optimizer = optim.Adam( model.parameters(), lr=0.001)
net, criterion, optimizer = get_model2(model, learning_rate=0.001, weight_decay=5e-4)

all_G_kernels = [
    Variable(kernel.data.clone(), requires_grad=True)
    for kernel in optimizer.param_groups[1]['params']
]
all_W_kernels = [kernel for kernel in optimizer.param_groups[1]['params']]
kernels = [{'params': all_G_kernels}]
optimizer_quant = optim.Adam(kernels, lr=0)
eta_rate = 1.05
eta = 1
# scheduler = MultiStepLR(optimizer=optimizer, milestones=[10,30,60,100,150,200], gamma=0.5)
scheduler = MultiStepLR(optimizer=optimizer, milestones=[10,60,100,150,200], gamma=0.5)
#####1:lr:0.0001 stride:[40,70,100,150,200],gamma = 0.2/0.5
#####2:lr:0.0005 stride:20,70,100,140,170,200
#####3:lr:0.0005 stride:10,30,60,100,150,200
#####4:lr:0.0005 stride:10,80,100,150,200

# 4. 创建writer
writer = SummaryWriter(log_dir="../weather_recognition/log/JZ/out34_lr1_dataall_smooth", flush_secs=500)
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
    RELOAD_CHECKPOINT_PATH = "../weather_recognition/model/JZ/JZ_transform4_lr1_0.001_deletee_gamma0.5_out_layer40.8590405904059041.pth"
    if RELOAD_CHECKPOINT == 1:
        PATH_TO_PTH_CHECKPOINT_G = torch.load(RELOAD_CHECKPOINT_PATH,map_location=lambda storage, loc: storage).state_dict()
        model.load_state_dict(PATH_TO_PTH_CHECKPOINT_G)
        
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
        label = smooth_label(label,8,0.1)
        batchAcc = 0  # 单批次正确率
        batchCorrectNum = 0  # 单批次正确个数
        all_W_kernels = optimizer.param_groups[1]['params']
        all_G_kernels = optimizer_quant.param_groups[0]['params']
        
        for i in range(len(all_W_kernels)):
            k_W = all_W_kernels[i]
            k_G = all_G_kernels[i]
            V = k_W.data

            #####Binary Connect#########################
            #k_G.data = quantize_bw(V)
            ############################################

            ######Binary Relax##########################
            if epoch<120:
                k_G.data = (eta*quantize_bw(V)+V)/(1+eta)

            else:
                k_G.data = quantize_bw(V)
            #############################################

            k_W.data, k_G.data = k_G.data, k_W.data
        optimizer.zero_grad()  # 清空梯度
        output = model(data)  # 获取模型输出
        # label = label.long()
        loss = torch.nn.functional.binary_cross_entropy(output, label)  # 计算损失
        
        # loss = torch.sum(-torch.sum(torch.mul(torch.log(torch.softmax(output, dim=-1)), label), dim=-1),dim=0) / 8.0##手动计算
        loss.backward()  # 反向传播梯度
        for i in range(len(all_W_kernels)):
            k_W = all_W_kernels[i]
            k_G = all_G_kernels[i]
            k_W.data, k_G.data = k_G.data, k_W.data
            
        optimizer.step()  # 更新参数
        scheduler.step()
        epochLoss += loss.item()  # 计算损失之和
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
    epochLoss = epochLoss  # 平均损失
    epochAcc = correctNum / len(trainLoader.dataset)  # 正确率
    print("Epoch:{}\t Loss:{} \t Acc:{}".format(epoch, epochLoss, epochAcc))
    # writer.add_scalar("train_loss", epochLoss, epoch)  # 写入日志
    # writer.add_scalar("train_acc", epochAcc, epoch)  # 写入日志
    #----------------------------------------------------------------------
    # Testing
    #----------------------------------------------------------------------
    model.eval()
    for i in range(len(all_W_kernels)):
        k_W = all_W_kernels[i]
        k_quant = all_G_kernels[i]    
        k_W.data, k_quant.data = k_quant.data, k_W.data
    
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
            all_W_kernels = optimizer.param_groups[1]['params']
            all_G_kernels = optimizer_quant.param_groups[0]['params']
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
                # dataframe.to_csv("../weather_recognition/data/JZ/val_out4_lr1_dataall_smooth.csv", index=False)

                        
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
    for epoch in range(1,Train.epochs + 1):
        trainAcc = train(epoch)
        valAcc = val(epoch)
        if valAcc > maxAcc:
            maxAcc = valAcc
            # 保存最大模型
            torch.save(model, Train.modelDir + "JZ_" + "out34_lr1_dataall_smooth_"  +str(valAcc) + ".pth")
    # 保存模型
    torch.save(model,Train.modelDir+ "JZ_"+ "out34_lr1_dataall_smooth_" + str(valAcc) + ".pth")



