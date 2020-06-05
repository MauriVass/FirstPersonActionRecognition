import torch
import resnetMod
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from MyConvLSTMCell import *


class attentionModel(nn.Module):
    def __init__(self, num_classes=61, mem_size=512, cam=True, ms=False, ms_task="classifier"):
        super(attentionModel, self).__init__()
        self.num_classes = num_classes
        self.resNet = resnetMod.resnet34(True, True)
        self.mem_size = mem_size
        self.weight_softmax = self.resNet.fc.weight
        self.lstm_cell = MyConvLSTMCell(512, mem_size)
        self.avgpool = nn.AvgPool2d(7)
        self.dropout = nn.Dropout(0.7)
        self.fc = nn.Linear(mem_size, self.num_classes)
        self.classifier = nn.Sequential(self.dropout, self.fc)
        self.compute_cam = cam
        self.include_ms = ms
        if ms:
            self.ms_conv = nn.Sequential(nn.ReLU(), nn.Conv2d(512, 100, 1))
            if ms_task == "classifier":  # ms_task either classifier or regressor
                self.ms_classifier = nn.Sequential(nn.Linear(4900, 98))
            else:  # regressor
                self.ms_classifier = nn.Sequential(nn.Linear(4900, 49))
                # 4900 = 100*7*7, 98 = 2*7*7
                # for classification, fc outputs 2*49 (49 samples, 2 classes) and forwards it to softmax to compute confidence scores (or does it?)
                # for regression, fc outputs the predictions straight away (49 samples)




    def forward(self, inputVariable):
        state = (Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((inputVariable.size(1), self.mem_size, 7, 7)).cuda()))
        ms_outputs = []
        for t in range(inputVariable.size(0)):
            logit, feature_conv, feature_convNBN = self.resNet(inputVariable[t])
            if self.compute_cam:
                bz, nc, h, w = feature_conv.size()
                feature_conv1 = feature_conv.view(bz, nc, h*w)
                probs, idxs = logit.sort(1, True)
                class_idx = idxs[:, 0]
                cam = torch.bmm(self.weight_softmax[class_idx].unsqueeze(1), feature_conv1)
                attentionMAP = F.softmax(cam.squeeze(1), dim=1)
                attentionMAP = attentionMAP.view(attentionMAP.size(0), 1, 7, 7)
                attentionFeat = feature_convNBN * attentionMAP.expand_as(feature_conv)
                state = self.lstm_cell(attentionFeat, state)
            else:
                state = self.lstm_cell(feature_convNBN, state)
            if self.include_ms:
                if self.compute_cam:
                    ms_out = self.ms_conv(attentionFeat)
                else:
                    ms_out = self.ms_conv(feature_conv)
                ms_out = torch.flatten(ms_out, 1)
                ms_out = self.ms_classifier(ms_out)
                ms_out = ms_out.view(inputVariable.size(1), 2, 7, 7)  # ms_out = ms_out.view(inputVariable.size(1), 49, 2)
                # ms_out = F.softmax(ms_out, dim=2)
                ms_outputs.append(ms_out)

        ms_outputs = torch.stack(ms_outputs, 0) # sape is seq_len x BSx2x7x7, is a good idea to stack over seq_len?
        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)
        if self.include_ms:
            return feats, feats1, ms_outputs
