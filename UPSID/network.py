import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from Attention import *
from LSTM import *


class RBNet(nn.Module):
    def __init__(self, recurrent_iter=6):
        super(RBNet, self).__init__()
        self.iteration = recurrent_iter

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32+32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32+32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def forward(self, input):
        batch_size, H, W = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, H, W))
        c = Variable(torch.zeros(batch_size, 32, H, W))

        h = h.cuda()
        c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + input

            x_list.append(x)

        return x, x_list


class Base_line(nn.Module):
    def __init__(self, recurrent_iter=6):
        super(Base_line, self).__init__()
        self.iteration = recurrent_iter

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32+32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32+32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def forward(self, input):
        batch_size, H, W = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, H, W))
        c = Variable(torch.zeros(batch_size, 32, H, W))

        h = h.cuda()
        c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h

            x = self.res_conv1(x)

            x = self.res_conv2(x)

            x = self.res_conv3(x)

            x =self.res_conv4(x)

            x = self.res_conv5(x)
            x = self.conv(x)

            x = x + input

            x_list.append(x)

        return x, x_list


class RBNet_Dense_Attention(nn.Module):
    def __init__(self, recurrent_iter):
        super(RBNet_Dense_Attention, self).__init__()
        self.iteration = recurrent_iter
        self.PSA = ParallelPolarizedSelfAttention(channel=32)
        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32 * 2, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32 * 3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32 * 4, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32 * 5, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 * 2, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 * 2, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 * 2, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 * 2, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def forward(self, input):
        batch_size, H, W = input.size(0), input.size(2), input.size(3)

        x = input
        h = Variable(torch.zeros(batch_size, 32, H, W))
        c = Variable(torch.zeros(batch_size, 32, H, W))

        h = h.cuda()
        c = c.cuda()

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)

            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = self.PSA(h)
            resx = x
            x1 = self.PSA(self.res_conv1(x))
            x1 = torch.cat((resx, x1), 1)
            resx = x1

            x2 = self.PSA(self.res_conv2(x1))
            x2 = torch.cat((resx, x2), 1)
            resx = x2
            x3 = self.PSA(self.res_conv3(x2))
            x3 = torch.cat((resx, x3), 1)
            resx = x3
            x4 = self.PSA(self.res_conv4(x3))
            x4 = torch.cat((resx, x4), 1)
            x = self.res_conv5(x4)
            x = self.conv(x)
            x = x + input

            x_list.append(x)

        return x, x_list


class BaseBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(BaseBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.conv0 = nn.Conv2d(out_channel * 3, out_channel, kernel_size=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=3, dilation=3)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out = torch.cat((out1, out2, out3), dim=1)
        out = self.conv0(out)
        return out


class AttentionResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AttentionResBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.Channel_Attention_1 = ChannelAttention(32)
        self.Channel_Attention_2 = ChannelAttention(32)
        self.conv_1 = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=3, stride=1, padding=1)
        self.LeakyReLU = nn.LeakyReLU(0.2, True)
        self.ReLU = nn.ReLU()
        self.conv_2 = nn.Conv2d(self.out_channel, self.out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        output = self.Channel_Attention_1(self.ReLU(self.conv_1(input)))
        # x = self.LeakyReLU(self.Channel_Attention_1(self.conv_1(input)))
        # output = self.LeakyReLU(self.Channel_Attention_2(self.conv_2(x)))
        return output


class RBNet_LSTM_Rain(nn.Module):
    def __init__(self, recurrent_iter=6):
        super(RBNet_LSTM_Rain, self).__init__()
        self.iteration = recurrent_iter

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )

        self.LSTM1 = LSTM(32)
        self.LSTM2 = LSTM(32)
        self.LSTM3 = LSTM(32)
        self.LSTM4 = LSTM(32)
        self.LSTM5 = LSTM(32)

        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def forward(self, input):
        batch_size, H, W = input.size(0), input.size(2), input.size(3)

        x = input
        h1, c1, h2, c2, h3, c3, h4, c4, h5, c5 = None, None, None, None, None, None, None, None, None, None


        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            h1, c1 = self.LSTM1(x, h1, c1)
            x = h1
            resx = x
            x = F.relu(self.res_conv1(x) + resx)

            h2, c2 = self.LSTM2(x, h2, c2)
            x = h2
            resx = x
            x = F.relu(self.res_conv2(x) + resx)

            h3, c3 = self.LSTM3(x, h3, c3)
            x = h3
            resx = x
            x = F.relu(self.res_conv3(x) + resx)

            h4, c4 = self.LSTM4(x, h4, c4)
            x = h4
            resx = x
            x = F.relu(self.res_conv4(x) + resx)

            h5, c5 = self.LSTM5(x, h5, c5)
            x = h5
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)
            x = x + input

            x_list.append(x)

        return x, x_list


class RBNet_LSTM_Atten(nn.Module):
    def __init__(self, recurrent_iter=6, channel=32):
        super(RBNet_LSTM_Atten, self).__init__()

        self.channel = channel
        self.iteration = recurrent_iter

        self.layer1 = AttentionResBlock(32, self.channel)
        self.layer2 = AttentionResBlock(32, self.channel)
        self.layer3 = AttentionResBlock(32, self.channel)
        self.layer4 = AttentionResBlock(32, self.channel)
        self.layer5 = AttentionResBlock(32, self.channel)

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )

        self.LSTM1 = LSTM(32)
        self.LSTM2 = LSTM(32)
        self.LSTM3 = LSTM(32)
        self.LSTM4 = LSTM(32)
        self.LSTM5 = LSTM(32)

        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def forward(self, input):
        batch_size, H, W = input.size(0), input.size(2), input.size(3)

        x = input
        h1, c1, h2, c2, h3, c3, h4, c4, h5, c5 = None, None, None, None, None, None, None, None, None, None


        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            h1, c1 = self.LSTM1(x, h1, c1)
            x = h1
            resx = x
            x = F.relu(self.layer1(x) + resx)

            h2, c2 = self.LSTM2(x, h2, c2)
            x = h2
            resx = x
            x = F.relu(self.layer2(x) + resx)

            h3, c3 = self.LSTM3(x, h3, c3)
            x = h3
            resx = x
            x = F.relu(self.layer3(x) + resx)

            h4, c4 = self.LSTM4(x, h4, c4)
            x = h4
            resx = x
            x = F.relu(self.layer4(x) + resx)

            h5, c5 = self.LSTM5(x, h5, c5)
            x = h5
            resx = x
            x = F.relu(self.layer5(x) + resx)
            x = self.conv(x)
            x = x + input

            x_list.append(x)

        return x, x_list



class RBNet_L_A_C_R(nn.Module):
    def __init__(self, recurrent_iter=6, channel=32):
        super(RBNet_L_A_C_R, self).__init__()

        self.channel = channel
        self.iteration = recurrent_iter

        self.layer1 = AttentionResBlock(32, self.channel)
        self.layer2 = AttentionResBlock(32, self.channel)
        self.layer3 = AttentionResBlock(32, self.channel)
        self.layer4 = AttentionResBlock(32, self.channel)
        self.layer5 = AttentionResBlock(32, self.channel)

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )

        self.LSTM1 = LSTM(32)
        self.LSTM2 = LSTM(32)
        self.LSTM3 = LSTM(32)
        self.LSTM4 = LSTM(32)
        self.LSTM5 = LSTM(32)

        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def forward(self, input):
        batch_size, H, W = input.size(0), input.size(2), input.size(3)

        x = input
        h1, c1, h2, c2, h3, c3, h4, c4, h5, c5 = None, None, None, None, None, None, None, None, None, None


        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            h1, c1 = self.LSTM1(x, h1, c1)
            x = h1
            resx = x
            x1 = F.relu(self.layer1(x) + resx)

            h2, c2 = self.LSTM2(x1, h2, c2)
            x = h1 + h2
            resx = x
            x2 = F.relu(self.layer2(x) + resx)

            h3, c3 = self.LSTM3(x2, h3, c3)
            x = h1 + h2 + h3
            resx = x
            x = F.relu(self.layer3(x) + resx)

            h4, c4 = self.LSTM4(x, h4, c4)
            x = h1 + h2 + h3 + h4
            resx = x
            x = F.relu(self.layer4(x) + resx)

            h5, c5 = self.LSTM5(x, h5, c5)
            x = h1 + h2 + h3 + h4 + h5
            resx = x
            x = F.relu(self.layer5(x) + resx)
            x = self.conv(x)
            x = x + input

            x_list.append(x)

        return x, x_list


class RBNet_LACR_Dense(nn.Module):
    def __init__(self, recurrent_iter=6, channel=32):
        super(RBNet_LACR_Dense, self).__init__()

        self.channel = channel
        self.iteration = recurrent_iter

        self.layer1 = AttentionResBlock(32, self.channel)
        self.layer2 = AttentionResBlock(32, self.channel)
        self.layer3 = AttentionResBlock(32, self.channel)
        self.layer4 = AttentionResBlock(32, self.channel)
        self.layer5 = AttentionResBlock(32, self.channel)

        self.conv_1 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        self.conv_2 = nn.Conv2d(96, 32, kernel_size=1, bias=False)
        self.conv_3 = nn.Conv2d(128, 32, kernel_size=1, bias=False)
        self.conv_4 = nn.Conv2d(160, 32, kernel_size=1, bias=False)


        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )

        self.LSTM1 = LSTM(32)
        self.LSTM2 = LSTM(32)
        self.LSTM3 = LSTM(32)
        self.LSTM4 = LSTM(32)
        self.LSTM5 = LSTM(32)

        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def forward(self, input):
        batch_size, H, W = input.size(0), input.size(2), input.size(3)

        x = input
        h1, c1, h2, c2, h3, c3, h4, c4, h5, c5 = None, None, None, None, None, None, None, None, None, None


        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            h1, c1 = self.LSTM1(x, h1, c1)
            x = h1
            resx = x
            x = F.relu(self.layer1(x) + resx)

            h2, c2 = self.LSTM2(x, h2, c2)
            x =torch.cat((h2, resx), 1)
            resx = x
            x = F.relu(self.layer2(self.conv_1(x)) + h2)

            h3, c3 = self.LSTM3(x, h3, c3)
            x = torch.cat((h3, resx), 1)
            resx = x
            x = F.relu(self.layer3(self.conv_2(x)) + h3)

            h4, c4 = self.LSTM4(x, h4, c4)
            x =torch.cat((h4, resx), 1)
            resx = x
            x = F.relu(self.layer4(self.conv_3(x)) + h4)

            h5, c5 = self.LSTM5(x, h5, c5)
            x = torch.cat((h5, resx), 1)
            resx = x
            x = F.relu(self.layer5(self.conv_4(x)) + h5)
            x = self.conv(x)
            x = x + input

            x_list.append(x)

        return x, x_list


class RBNet_LACR_Dense_new(nn.Module):
    def __init__(self, recurrent_iter=6, channel=32):
        super(RBNet_LACR_Dense_new, self).__init__()

        self.channel = channel
        self.iteration = recurrent_iter
        self.Channel_Attention_1 = ChannelAttention(32)
        self.Channel_Attention_2 = ChannelAttention(32)
        self.Channel_Attention_3 = ChannelAttention(32)
        self.Channel_Attention_4 = ChannelAttention(32)
        self.Channel_Attention_5 = ChannelAttention(32)

        # self.layer0 = AttentionResBlock(32, self.channel)
        self.layer1 = AttentionResBlock(32, self.channel)
        self.layer2 = AttentionResBlock(32, self.channel)
        self.layer3 = AttentionResBlock(32, self.channel)
        self.layer4 = AttentionResBlock(32, self.channel)
        self.layer5 = AttentionResBlock(32, self.channel)

        self.conv_1 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        self.conv_2 = nn.Conv2d(96, 32, kernel_size=1, bias=False)
        self.conv_3 = nn.Conv2d(128, 32, kernel_size=1, bias=False)
        self.conv_4 = nn.Conv2d(160, 32, kernel_size=1, bias=False)
        self.conv_5 = nn.Conv2d(192, 32, kernel_size=1, bias=False)


        self.conv_in = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )

        self.LSTM1 = LSTM(32)
        self.LSTM2 = LSTM(32)
        self.LSTM3 = LSTM(32)
        self.LSTM4 = LSTM(32)
        self.LSTM5 = LSTM(32)

        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU()
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def forward(self, input):
        batch_size, H, W = input.size(0), input.size(2), input.size(3)

        x = input
        h1, c1, h2, c2, h3, c3, h4, c4, h5, c5 = None, None, None, None, None, None, None, None, None, None


        x_list = []
        for i in range(self.iteration):
            # block_0: input_channel=6
            x = torch.cat((input, x), 1)
            x = self.conv_in(x)
            resx = x

            # the first internal and external dense connection block
            h1, c1 = self.LSTM1(x, h1, c1)
            # x = h1
            x = torch.cat((h1, resx), 1)
            resx = x
            # x = F.relu(self.layer1(x) + resx)
            # x = self.conv1(self.Channel_Attention_1(self.conv_1(x)) + h1)
            # x = self.conv1(torch.cat((self.Channel_Attention_1(self.conv_1(x)), h1), 1))
            x = self.conv1(torch.cat((self.layer1(self.conv_1(x)), h1), 1))

            h2, c2 = self.LSTM2(x, h2, c2)
            x =torch.cat((h2, resx), 1)
            resx = x
            # x = self.conv2(torch.cat((self.Channel_Attention_2(self.conv_2(x)), h2), 1))
            x = self.conv2(torch.cat((self.layer2(self.conv_2(x)), h2), 1))

            h3, c3 = self.LSTM3(x, h3, c3)
            x = torch.cat((h3, resx), 1)
            resx = x
            x = self.conv3(torch.cat((self.layer3(self.conv_3(x)), h3), 1))

            h4, c4 = self.LSTM4(x, h4, c4)
            x =torch.cat((h4, resx), 1)
            resx = x
            x = self.conv4(torch.cat((self.layer4(self.conv_4(x)), h4), 1))

            h5, c5 = self.LSTM5(x, h5, c5)
            x = torch.cat((h5, resx), 1)
            # resx = x
            x = self.conv5(torch.cat((self.layer5(self.conv_5(x)), h5), 1))

            x = self.conv_out(x)
            x = x + input

            x_list.append(x)

        return x, x_list


class RBNet_LACR_Dense_final(nn.Module):
    def __init__(self, recurrent_iter=6, channel=32):
        super(RBNet_LACR_Dense_final, self).__init__()

        self.channel = channel
        self.iteration = recurrent_iter
        self.Channel_Attention_1 = ChannelAttention(32)
        self.Channel_Attention_2 = ChannelAttention(32)
        self.Channel_Attention_3 = ChannelAttention(32)
        self.Channel_Attention_4 = ChannelAttention(32)
        self.Channel_Attention_5 = ChannelAttention(32)

        # self.layer0 = AttentionResBlock(32, self.channel)
        self.layer1 = AttentionResBlock(32, self.channel)
        self.layer2 = AttentionResBlock(32, self.channel)
        self.layer3 = AttentionResBlock(32, self.channel)
        self.layer4 = AttentionResBlock(32, self.channel)
        self.layer5 = AttentionResBlock(32, self.channel)

        self.conv_1 = nn.Conv2d(64, 32, kernel_size=1, bias=False)
        self.conv_2 = nn.Conv2d(96, 32, kernel_size=1, bias=False)
        self.conv_3 = nn.Conv2d(128, 32, kernel_size=1, bias=False)
        self.conv_4 = nn.Conv2d(160, 32, kernel_size=1, bias=False)
        self.conv_5 = nn.Conv2d(192, 32, kernel_size=1, bias=False)


        self.conv_in = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )

        self.LSTM1 = LSTM(32)
        self.LSTM2 = LSTM(32)
        self.LSTM3 = LSTM(32)
        self.LSTM4 = LSTM(32)
        self.LSTM5 = LSTM(32)

        self.conv1 = nn.Sequential(
            nn.Conv2d(96, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(160, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(192, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(224, 32, 3, 1, 1),
            nn.ReLU()
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def forward(self, input):
        batch_size, H, W = input.size(0), input.size(2), input.size(3)

        x = input
        h1, c1, h2, c2, h3, c3, h4, c4, h5, c5 = None, None, None, None, None, None, None, None, None, None


        x_list = []
        for i in range(self.iteration):
            # block_0: input_channel=6
            x = torch.cat((input, x), 1)
            x = self.conv_in(x)
            resx = x

            # the first internal and external dense connection block
            h1, c1 = self.LSTM1(x, h1, c1)
            # x = h1
            x = torch.cat((h1, resx), 1)
            resx = x
            # x = F.relu(self.layer1(x) + resx)
            # x = self.conv1(self.Channel_Attention_1(self.conv_1(x)) + h1)
            # x = self.conv1(torch.cat((self.Channel_Attention_1(self.conv_1(x)), h1), 1))
            # x = self.conv1(torch.cat((self.layer1(self.conv_1(x)), h1), 1))
            x = self.conv1(torch.cat((self.layer1(self.conv_1(x)), x), 1))

            h2, c2 = self.LSTM2(x, h2, c2)
            x =torch.cat((h2, resx), 1)
            resx = x
            # x = self.conv2(torch.cat((self.Channel_Attention_2(self.conv_2(x)), h2), 1))
            # x = self.conv2(torch.cat((self.layer2(self.conv_2(x)), h2), 1))
            x = self.conv2(torch.cat((self.layer2(self.conv_2(x)), x), 1))

            h3, c3 = self.LSTM3(x, h3, c3)
            x = torch.cat((h3, resx), 1)
            resx = x
            # x = self.conv3(torch.cat((self.layer3(self.conv_3(x)), h3), 1))
            x = self.conv3(torch.cat((self.layer3(self.conv_3(x)), x), 1))

            h4, c4 = self.LSTM4(x, h4, c4)
            x =torch.cat((h4, resx), 1)
            resx = x
            # x = self.conv4(torch.cat((self.layer4(self.conv_4(x)), h4), 1))
            x = self.conv4(torch.cat((self.layer4(self.conv_4(x)), x), 1))

            h5, c5 = self.LSTM5(x, h5, c5)
            x = torch.cat((h5, resx), 1)
            # resx = x
            # x = self.conv5(torch.cat((self.layer5(self.conv_5(x)), h5), 1))
            x = self.conv5(torch.cat((self.layer5(self.conv_5(x)), x), 1))

            x = self.conv_out(x)
            x = x + input

            x_list.append(x)

        return x, x_list

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = RBNet_Dense_Attention(recurrent_iter=6)
    input = torch.randn((4, 3, 64, 64))
    input = input.cuda()
    model = model.cuda()
    output, out_list = model(input)
    print(output.size(), len(out_list))
