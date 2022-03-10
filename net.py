import torch
from torch import nn
import numpy as np
import os
import cv2

class Modle(nn.Module):

    def __init__(self):
        super(Modle, self).__init__()
        self.zeropad = nn.ZeroPad2d(padding=(2,2,2,2))
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.softmax_2 = nn.Softmax2d()
        self.softmax_1 = nn.Softmax()

        self.conv_1 = nn.Conv2d(in_channels=4, out_channels=50, kernel_size=(5,5),padding=2, stride=2)
        self.batch_normal_1 = nn.BatchNorm2d(50)
        self.maxpool = nn.MaxPool2d((2,2))

        self.conv_2 = nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(5, 5), padding=2, stride=2)
        self.batch_normal_2 = nn.BatchNorm2d(100)

        self.conv_3 = nn.Conv2d(in_channels=100, out_channels=150, kernel_size=(5, 5), padding=2, stride=2)
        self.batch_normal_3 = nn.BatchNorm2d(150)

        self.conv_4 = nn.Conv2d(in_channels=150, out_channels=200, kernel_size=(5, 5), padding=2, stride=2)
        self.batch_normal_4 = nn.BatchNorm2d(200)

        self.linear_1 = nn.Linear(in_features=3200, out_features=600)
        self.batch_normal_5 = nn.BatchNorm1d(600)

        self.linear_2 = nn.Linear(in_features=600, out_features=400)
        self.batch_normal_6 = nn.BatchNorm1d(400)

        self.linear_3 = nn.Linear(in_features=400, out_features=4)


        # self.linear_4 = nn.Linear(in_features=3200, out_features=600)
        # self.batch_normal_5 = nn.BatchNorm1d(3200)


    def forward(self, input):

        x = self.zeropad(input)

        x = self.conv_1(x)
        x = self.relu(x)
        x = self.batch_normal_1(x)
        x = self.maxpool(x)

        x = self.conv_2(x)
        x = self.relu(x)
        x = self.batch_normal_2(x)
        x = self.dropout(x)

        x = self.conv_3(x)
        x = self.relu(x)
        x = self.batch_normal_3(x)
        x = self.dropout(x)

        x = self.conv_4(x)
        x = self.relu(x)
        x = self.batch_normal_4(x)
        x = self.dropout(x)

        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))

        x = self.relu(self.linear_1(x))
        x = self.batch_normal_5(x)

        x = self.relu(self.linear_2(x))
        x = self.batch_normal_6(x)
        x = self.dropout(x)

        x = self.linear_3(x)

        # x = x.view(4,4)
        x = self.softmax_1(x)
        x = 6* x
        # x_2 = self.softmax_2(x)
        return x






# # 多增加1个维度
# def add_dim(img_list):
# # np.expand_dims(img_list[0], axis=0).shape
#     input = np.expand_dims(img_list[0], axis=0)
#     for i in list(img_list)[1:]:
#         # 增加维度
#         i = np.expand_dims(i, axis=0)
#         # 拼接
#         input = np.concatenate((input, i), axis=0)
#
#     # 变换顺序
#     # input = np.transpose(input, (0,3,1,2))
#     return input
#
# # 图片分割
# def img_split(image):
#     first = image[:100,:100]
#     second = image[:100,100:]
#     third = image[100:,:100]
#     fourth = image[100:,100:]
#
#     return first, second, third, fourth
#
#
# # 读图
# path = './data/puzzle_2x2/test'
# dir_con = os.listdir(path)
# img_path = [os.path.join(path,img) for img in dir_con]
# first_img = cv2.imread(img_path[0],0)
# img_list = img_split(first_img)
# aa = add_dim(img_list)
#
# input = np.expand_dims(aa, axis=0)
# input = np.concatenate((input,input), axis=0)
#
# input = torch.from_numpy(input)
# input = input.type(torch.float32)
# model = Modle()
#
# model.eval()
# pre = model(input)
#
#
# print(pre)