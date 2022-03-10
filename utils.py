import os
import numpy as np
import cv2
import pandas as pd
from torch.utils import data
import torch
import time
from net import Modle
from torch import nn


# 创建dataset 的子类
class Mydataset(data.Dataset):
    def __init__(self, imgs_path, labels ):
        self.imgs_path = imgs_path
        self.labels = labels

    def __getitem__(self, index):
        img = self.imgs_path[index]
        img = self.read_img(img)
        img = torch.tensor(img, dtype=torch.float32)
        label = self.labels[index]

        return img,label

    def __len__(self):
        return len(self.imgs_path)


    def read_img(self, path):
        read_img = cv2.imread(path,0)
        img = self.preprocess(read_img)
        img = self.add_dim(img)
        return img


    def preprocess(self, image):
        first = image[:100,:100]
        second = image[:100,100:]
        third = image[100:,:100]
        fourth = image[100:,100:]
        return first, second, third, fourth

    def add_dim(self, img_list):
        # np.expand_dims(img_list[0], axis=0).shape
        input = np.expand_dims(img_list[0], axis=0)
        for i in list(img_list)[1:]:
            # 增加维度
            i = np.expand_dims(i, axis=0)
            # 拼接
            input = np.concatenate((input, i), axis=0)

        # 变换顺序
        # input = np.transpose(input, (0,3,1,2))
        return input



def read_data(path,type):
    # 拼接路径
    train_img = os.path.join(path, type)
    train_label = os.path.join(path, type+ '.csv')

    # 获取图片的path列表
    train_img_path = os.listdir(train_img)
    train_img_path.sort(key=lambda x:int(x.split('.')[0]))
    train_img_path = list(map(lambda x:os.path.join(train_img,x),train_img_path))

    # 处理标签数据
    labels = pd.read_csv(train_label)
    labels = labels.label.apply(lambda x:np.array([int(i) for i in x.split()]).reshape(1,-1))
    labels = labels.values

    first_label = labels[0]
    other_label = labels[1:]
    for i in other_label:
        first_label = np.concatenate((first_label,i), axis=0)

    first_label = torch.tensor(first_label,dtype=torch.float32)

    return train_img_path, first_label


def decode(pre_data):

    b = np.array(range(4))
    c = []
    d = []
    # 按照a的大小顺序对b进行排序
    for i in np.lexsort((b, pre_data.data)):
        c.append(b[i])

    for i in np.lexsort((b, c)):
        d.append(b[i])
    return np.array(d).reshape(1,-1)

# a = np.array([0.1,2.5,1.5,1.9])
# decode(a)

def pre2label(pre):
    first = decode(pre[0])
    for k in pre[1:]:
        first = np.concatenate((first,decode(k)),axis=0)
    return first

# labels_batch
# a = pre2label(pre)
def acc(pre,label):
    pre = pre2label(pre)
    return ((pre == label.numpy()).sum(1)==4).mean()

# acc(pre,labels_batch)


def img_recover(img_list, label):
    index = range(0,4)
    index_1 = np.lexsort((index, label))
    sort = [index[i] for i in index_1]
    heng_1 = np.concatenate((img_list[sort[0]],img_list[sort[1]]),axis=1)
    heng_2 = np.concatenate((img_list[sort[2]],img_list[sort[3]]),axis=1)
    fl = np.concatenate((heng_1,heng_2), axis=0)
    return fl



def train(train_dataloader,
          test_dataloader,
          model,
          loss_fn,
          optim,
          epoches=50,
          device = ('cuda:0' if torch.cuda.is_available() else 'cpu')):

    model.to(device)
    for epoch in range(epoches):
        # 初始化
        train_acc = 0
        train_loss = 0
        test_acc = 0
        test_loss = 0
        start = time.time()

        # 设置pytorch的训练模式drop_out发挥作用
        model.train()
        for x, y in train_dataloader:
            # 将数据集转移到gpu
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            optim.zero_grad()
            loss = loss_fn(y_pred, y)
            loss.backward()
            optim.step()

            with torch.no_grad():
                # 计算正确率与损失
                train_acc = train_acc + acc(y_pred.cpu(), y.cpu())
                train_loss = train_loss + loss.data.item()
                # print(train_acc)

        # 预测模式，drop_out不发挥作用 主要影响drop_out 与 BN层
        model.eval()
        with torch.no_grad():
            for x, y in test_dataloader:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)
                loss = loss_fn(y_pred, y).data.item()
                test_acc = test_acc + acc(y_pred.cpu(), y.cpu())
                test_loss = test_loss + loss

            end = time.time()
            # 计算平均值
            train_loss = train_loss / len(train_dataloader)
            train_acc = train_acc / len(train_dataloader)

            test_loss = test_loss / len(test_dataloader)
            test_acc = test_acc / len(test_dataloader)
            print('当前epoch为:{},训练集损失为:{},训练集正确率为:{},验证集损失为:{},验证集正确率为:{},用时:{}s'.format(epoch,
                                                                                        train_loss,
                                                                                        train_acc,
                                                                                        test_loss,
                                                                                        test_acc,
                                                                                        end - start))

    return model


def valid(valid_dataloader,
          model,
          loss_fn,
          device = ('cuda:0' if torch.cuda.is_available() else 'cpu')):

    model.to(device)
    valid_acc = 0
    valid_loss = 0
    i = 0
    model.eval()

    for x, y in valid_dataloader:

        i += 1
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = loss_fn(y_pred, y).data.item()
        valid_acc = valid_acc + acc(y_pred.cpu(), y.cpu())
        valid_loss = valid_loss + loss

        if i == 1:
            valid_data = x.cpu().data.numpy()
            valid_label = y.cpu().data.numpy()
            valid_pre = pre2label(y_pred.cpu())

        # 合并
        valid_data = np.concatenate((valid_data, x.cpu().data.numpy()), axis=0)
        valid_label = np.concatenate((valid_label, y.cpu().data.numpy()), axis=0)
        # valid_pre = np.concatenate((valid_pre, y_pred.cpu().data.numpy()), axis=0)
        valid_pre = np.concatenate((valid_pre, pre2label(y_pred.cpu())), axis=0)



    valid_loss = valid_loss / len(valid_dataloader)
    valid_acc = valid_acc / len(valid_dataloader)

    print('valid数据集损失为:{},valid数据集正确率为:{}'.format(valid_loss,valid_acc))

    return valid_loss,valid_acc,valid_data,valid_label,valid_pre

# 恢复图片
def img_recover(img, label):
    img_list = [img[0],img[1],img[2],img[3]]
    index = range(0,4)
    index_1 = np.lexsort((index, label))
    sort = [index[i] for i in index_1]
    heng_1 = np.concatenate((img_list[sort[0]],img_list[sort[1]]),axis=1)
    heng_2 = np.concatenate((img_list[sort[2]],img_list[sort[3]]),axis=1)
    fl = np.concatenate((heng_1,heng_2), axis=0)
    return fl


# # 在valid数据集验证
# path_valid = './data/puzzle_2x2/'
# valid_img,valid_label = read_data(path_valid,type='valid')
#
# # valid数据集
# valid_dataset = Mydataset(valid_img,valid_label)
# # 创建dataloader
# valid_dataloader = data.DataLoader(dataset=valid_dataset,
#                                    batch_size=256,)
#
#
# # 加载模型
# valid_model = Modle()
# valid_loss_fn = nn.CrossEntropyLoss()
# valid_model.load_state_dict(torch.load('./model/model.pkl'))
# print(valid(model=valid_model,valid_dataloader=valid_dataloader,loss_fn=valid_loss_fn))



