import torch
from flow_resnet import *
from objectAttentionModelConvLSTM import *
import torch.nn as nn


class twoStreamAttentionModel(nn.Module):
    def __init__(self, flow_model_path=None, rgb_model_path=None, seq_len_flow=5, mem_size=512, num_classes=61, join_method="average", ordering=False):
        super(twoStreamAttentionModel, self).__init__()
        self.join_method = join_method
        self.flow_model = flow_resnet34(False, channels=2*seq_len_flow, num_classes=num_classes)
        if flow_model_path is not None:
            self.flow_model.load_state_dict(torch.load(flow_model_path))
        self.rgb_model = attentionModel(num_classes, mem_size,ordering=ordering)
        if rgb_model_path is not None:
            self.rgb_model.load_state_dict(torch.load(rgb_model_path))
        self.fc2 = nn.Linear(512 * 2, num_classes, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Sequential(self.dropout, self.fc2)

    def forward(self, input_flow, input_rgb):
        rgb_output, rgb_features = self.rgb_model(input_rgb)
        flow_output, flow_features = self.flow_model(input_flow)
        if self.join_method == "joint":  # pass the concatenated features to FC classifier
            two_stream_features = torch.cat((flow_features, rgb_features), 1)
            return self.classifier(two_stream_features)
        else:  # average the scores
            average_output = 0.5*rgb_output + 0.5*flow_output
            return average_output
