import torch
import resnetMod
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from MyConvLSTMCell import *
from itertools import permutations
from numpy.random import randint
import numpy as np


def gen_permutations(length, seq_len, num_permutations):
    all_perm = np.array(list(permutations(range(seq_len), length)))
    return all_perm[randint(len(all_perm), size=num_permutations)]


def to_relative_order(perms):
    for perm in perms:
        order = np.argsort(perm)
        for i in range(len(order)):
            perm[[order[i]]] = i
    return perms



class attentionModel(nn.Module):
    def __init__(self, num_classes=61, mem_size=512, cam=True, ms=False, ms_task="classifier", ordering=True, order_type="rgb", perm_tuple_length=3, num_perms=10): # order_type "rgb" or "of"
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
        self.ms_task = ms_task
        self.include_ordering = ordering
        if ms:
            self.ms_conv = nn.Sequential(nn.ReLU(), nn.Conv2d(512, 100, 1))
            if self.ms_task == "classifier":  # ms_task either classifier or regressor
                self.ms_classifier = nn.Sequential(nn.Linear(4900, 98))
            else:  # regressor
                self.ms_classifier = nn.Sequential(nn.Linear(4900, 49))
                # 4900 = 100*7*7, 98 = 2*7*7
                # for classification, fc outputs 2*49 (49 samples, 2 classes) and forwards it to softmax to compute confidence scores (or does it?)
                # for regression, fc outputs the predictions straight away (49 samples)
        if self.include_ordering:
            self.ordering_lstm_cell = MyConvLSTMCell(512, mem_size)
            self.order_type = order_type
            self.perm_tuple_length = perm_tuple_length
            self.num_perms = num_perms
            self.order_classifier = nn.Sequential(nn.Dropout(0.7), nn.Linear(mem_size, self.perm_tuple_length))


    def forward_order(self, inputs):
        state = (Variable(torch.zeros((inputs.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((inputs.size(1), self.mem_size, 7, 7)).cuda()))
        batch_size = inputs.size(1)
        if self.order_type == "rgb":
            lstm_out_freq = 1
        else:
            lstm_out_freq = 2
        perm_out = []
        seq_len = inputs.size(0)
        perms = gen_permutations(self.perm_tuple_length, seq_len, self.num_perms)
        for perm in perms:
            perm_scores = []
            for index in perm:
                logit, feature_conv, feature_convNBN = self.resNet(inputs[index])
                state = self.ordering_lstm_cell(feature_convNBN, state)
                if (index+1) % lstm_out_freq == 0:
                    feats = self.avgpool(state[1]).view(state[1].size(0), -1)  # state[1] is BSx512x7x7, feats1 is BSx512
                    feats = self.order_classifier(feats)  # BS x perm_tuple_length
                    perm_scores.append(feats)
            perm_scores = torch.stack(perm_scores, dim=2)  # perm_scores is BSx3 (scores) x 3(perm index)
            perm_out.append(perm_scores)
        perm_out = torch.stack(perm_out, dim=3)  # BSx3x3x10
        perms = to_relative_order(perms)
        perms = torch.from_numpy(perms).repeat(batch_size, 1, 1).permute(0, 2, 1).to(torch.int64)
        return perms, perm_out  #  perms should be BSx1x3x10




    def forward(self, inputs, order_only=False):
        if order_only:
            return self.forward_order(inputs)
        state = (Variable(torch.zeros((inputs.size(1), self.mem_size, 7, 7)).cuda()),
                 Variable(torch.zeros((inputs.size(1), self.mem_size, 7, 7)).cuda()))
        ms_outputs = []
        for t in range(inputs.size(0)):
            logit, feature_conv, feature_convNBN = self.resNet(inputs[t])
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
                    ms_out = self.ms_conv(feature_conv)  # BSx100x7x7
                ms_out = torch.flatten(ms_out, 1)  # BSx4900
                ms_out = self.ms_classifier(ms_out)  # BSx98
                if self.ms_task == "classifier":
                    ms_out = ms_out.view(inputs.size(1), 2, 7, 7)  # ms_out = ms_out.view(inputVariable.size(1), 49, 2)  #  #BSX2x7x7
                else:
                    ms_out = ms_out.view(inputs.size(1), 1, 7, 7)
                # ms_out = F.softmax(ms_out, dim=2)
                ms_outputs.append(ms_out)
        feats1 = self.avgpool(state[1]).view(state[1].size(0), -1)
        feats = self.classifier(feats1)
        if self.include_ms:
            ms_outputs = torch.stack(ms_outputs, 2) # BSxseq_lenx2x7x7
            if self.include_ordering:
                order_labels, order_feats = self.forward_order(inputs)
                return feats, feats1, ms_outputs, order_labels, order_feats
            else:
                return feats, feats1, ms_outputs
        elif self.include_ordering:
            order_labels, order_feats = self.forward_order(inputs)
            return feats, feats1, order_labels, order_feats
        else:
            return feats, feats1
