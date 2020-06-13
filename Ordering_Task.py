import os
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader, random_split
from torch.backends import cudnn

import torchvision
from torchvision import transforms
from torchvision.models import resnet34

from PIL import Image
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from datetime import datetime
from utils import Config, test, display_map_prediction, FocalLoss
from gtea_dataset import gtea61
# %%
# !wandb login
# %%

config_stage2_ordering = Config({"stage": 2,
                                 "ms": True,
                                 "ordering": True,
                                 "order_type": "rgb",
                                 "perm_tuple_length": 3,
                                 "num_perms": 10,
                                 "ms_task": "classifier",  # ms_task is either classifier or regressor
                                 "ms_features": "CAM",  # RN: ResNet, CAM: attention
                                 "order_features": "RN",
                                 "binary_mask_threshold": 0.01,
                                 "num_classes": 61,
                                 "batch_size": 8,
                                 "lstm_mem_size": 512,
                                 "lr": 1e-4,
                                 "ms_lr": 1e-5,
                                 "ordering_lr": 1e-5,
                                 "optimizer": "adam",
                                 "epochs": 170,
                                 "decay_steps": [100, 150],
                                 "decay_factor": 0.1,
                                 "weight_decay": 5e-4,
                                 "val_frequency": 3,
                                 "models_dir": "models",
                                 "seq_len": 7,
                                 "training_user_split": [1, 3, 4],
                                 "val_user_split": [2],
                                 "cam": True})
#%%

def prepare_training_ms(config):
    train_params_rgb = []
    train_params_ms = []
    train_params_ordering = []

    model = attentionModel(num_classes=config.num_classes, mem_size=config.lstm_mem_size, cam=config.cam, ms=True, ms_task=config.ms_task,
                           ordering=config.ordering, order_type=config.order_type, perm_tuple_length=config.perm_tuple_length, num_perms=config.num_perms)
    stage1_dict = os.path.join(config.models_dir , 'best_model_rgb_state_dict.pth')
    model.load_state_dict(torch.load(stage1_dict), strict=False)
    model.train(False)
    for params in model.parameters():
        params.requires_grad = False
    #
    for params in model.resNet.layer4[0].conv1.parameters():
        params.requires_grad = True
        train_params_rgb += [params]

    for params in model.resNet.layer4[0].conv2.parameters():
        params.requires_grad = True
        train_params_rgb += [params]

    for params in model.resNet.layer4[1].conv1.parameters():
        params.requires_grad = True
        train_params_rgb += [params]

    for params in model.resNet.layer4[1].conv2.parameters():
        params.requires_grad = True
        train_params_rgb += [params]

    for params in model.resNet.layer4[2].conv1.parameters():
        params.requires_grad = True
        train_params_rgb += [params]
    #
    for params in model.resNet.layer4[2].conv2.parameters():
        params.requires_grad = True
        train_params_rgb += [params]
    #
    for params in model.resNet.fc.parameters():
        params.requires_grad = True
        train_params_rgb += [params]

    model.resNet.layer4[0].conv1.train(True)
    model.resNet.layer4[0].conv2.train(True)
    model.resNet.layer4[1].conv1.train(True)
    model.resNet.layer4[1].conv2.train(True)
    model.resNet.layer4[2].conv1.train(True)
    model.resNet.layer4[2].conv2.train(True)
    model.resNet.fc.train(True)

    model.ms_conv.train(True)
    model.ms_classifier.train(True)

    for params in model.ms_conv.parameters():
        params.requires_grad = True
        train_params_ms += [params]

    for params in model.ms_classifier.parameters():
        params.requires_grad = True
        train_params_ms += [params]

    for params in model.lstm_cell.parameters():
        params.requires_grad = True
        train_params_rgb += [params]

    for params in model.classifier.parameters():
        params.requires_grad = True
        train_params_rgb += [params]

    for params in model.ordering_lstm_cell.parameters():
        params.requires_grad = True
        train_params_ordering += [params]

    for params in model.order_classifier.parameters():
        params.requires_grad = True
        train_params_ordering += [params]

    return model, train_params_rgb, train_params_ms, train_params_ordering


def training_ms(model, config, train_loader, val_loader):
    wandb.watch(model, log="all")
    train_iter = 0
    best_accuracy = 0
    for epoch in range(config.epochs):
        epoch_loss_rgb = 0
        epoch_loss_ms = 0
        epoch_loss_ordering = 0
        num_corrects_rgb = 0
        num_corrects_ms = 0
        trainSamples = 0
        map_pixel_samples = 0
        iterPerEpoch = 0
        model.lstm_cell.train(True)
        model.classifier.train(True)
        model.resNet.layer4[0].conv1.train(True)
        model.resNet.layer4[0].conv2.train(True)
        model.resNet.layer4[1].conv1.train(True)
        model.resNet.layer4[1].conv2.train(True)
        model.resNet.layer4[2].conv1.train(True)
        model.resNet.layer4[2].conv2.train(True)
        model.resNet.fc.train(True)
        model.ms_conv.train(True)
        model.ms_classifier.train(True)
        # display_ms = True
        for inputs_rgb, map_labels, labels in train_loader:
            num_samples = inputs_rgb.size(0)
            trainSamples += num_samples
            train_iter += 1
            iterPerEpoch += 1
            optimizer_fn.zero_grad()
            inputs_rgb = inputs_rgb.permute(1, 0, 2, 3, 4).to(config.device)  # but why?
            labels = labels.to(config.device)
            # map_labels = map_labels.to(config.device)
            map_labels = map_labels.to(config.device).permute(0, 2, 1, 3, 4).squeeze() # BSxseq_lenx7x7
            output_label, _, output_map, order_labels, order_feats = model(inputs_rgb)  # output_map is BSx2
            # if display_ms:
            #     display_map_prediction(map_labels[0].clone(), output_map[0].clone(), loss_fn_ms, config.ms_task)
            #     display_ms = False
            map_pixel_samples += num_samples * config.seq_len * 7 * 7
            loss_rgb = loss_fn_rgb(output_label, labels)
            loss_ms = loss_fn_ms(output_map.squeeze(), map_labels)
            loss_ordering = loss_fn_ordering(order_feats, order_labels.to(config.device))
            # loss_ratio = loss_rgb.item()/loss_ms.item()
            loss = loss_rgb + loss_ms + loss_ordering
            loss.backward()
            optimizer_fn.step()

            _, predicted_rgb = torch.max(output_label.data, 1)
            epoch_loss_rgb += loss_rgb.item()
            predicted_rgb = predicted_rgb.to(config.device)
            num_corrects_rgb += torch.sum(predicted_rgb == labels).data.item()

            if config.ms_task == "classifier":
                _, predicted_ms = torch.max(output_map.data, 1)
                predicted_ms = predicted_ms.to(config.device)
                num_corrects_ms += torch.sum(predicted_ms == map_labels).data.item()
            epoch_loss_ms += loss_ms.item()
            epoch_loss_ordering += loss_ordering.item()


        optim_scheduler.step()
        avg_loss_rgb = epoch_loss_rgb / iterPerEpoch
        train_accuracy_rgb = (num_corrects_rgb / trainSamples)
        avg_loss_ms = epoch_loss_ms / iterPerEpoch
        avg_loss_ordering = epoch_loss_ordering / iterPerEpoch
        if config.ms_task == "classifier":
            train_accuracy_ms = (num_corrects_ms / map_pixel_samples)

        print('Train: Epoch = {}/{} | Loss = {} | Accuracy = {}'.format(epoch + 1, config.epochs, avg_loss_rgb, train_accuracy_rgb))

        max_loss = 6
        avg_loss_normalized_rgb = avg_loss_rgb if avg_loss_rgb < max_loss else max_loss
        avg_loss_normalized_ms = avg_loss_ms if avg_loss_ms < max_loss else max_loss
        if config.ms_task == "classifier":
            wandb.log({"train_loss_rgb": avg_loss_normalized_rgb,
                       "train_loss_ms": avg_loss_normalized_ms,
                       "train_accuracy_rgb": train_accuracy_rgb,
                       "train_accuracy_ms": train_accuracy_ms,
                       "avg_loss_ordering": avg_loss_ordering,
                       "eopch": (epoch + 1)})
        else:
            wandb.log({"train_loss_rgb": avg_loss_normalized_rgb,
                       "train_loss_ms": avg_loss_normalized_ms,
                       "train_accuracy_rgb": train_accuracy_rgb,
                       "avg_loss_ordering": avg_loss_ordering,
                       "eopch": (epoch + 1)})

        if (epoch + 1) % config.val_frequency == 0:
            with torch.no_grad():
                model.eval()
                val_loss_epoch_rgb = 0
                val_loss_epoch_ms = 0
                val_iter = 0
                val_samples = 0
                num_corrects_rgb = 0
                num_corrects_ms = 0
                map_pixel_samples = 0
                val_loss_epoch_ordering = 0
                for inputs_rgb, map_labels, labels in val_loader:
                    val_iter += 1
                    num_samples = inputs_rgb.size(0)
                    val_samples += num_samples
                    inputs_rgb = inputs_rgb.permute(1, 0, 2, 3, 4).to(config.device)
                    labels = labels.to(config.device)
                    map_labels = map_labels.to(config.device).view(num_samples, config.seq_len, 7, 7)
                    output_label, _, output_map, order_labels, order_feats = model(inputs_rgb)
                    map_pixel_samples += num_samples * config.seq_len * 7 * 7
                    val_loss_rgb = loss_fn_rgb(output_label, labels)
                    val_loss_ms = loss_fn_ms(output_map.squeeze(), map_labels)
                    loss_ordering = loss_fn_ordering(order_feats, order_labels.to(config.device))
                    val_loss_epoch_rgb += val_loss_rgb.item()
                    val_loss_epoch_ms += val_loss_ms.item()
                    val_loss_epoch_ordering += loss_ordering.item()
                    _, predicted_rgb = torch.max(output_label.data, 1)
                    num_corrects_rgb += torch.sum(predicted_rgb == labels).data.item()
                    if config.ms_task == "classifier":
                        _, predicted_ms = torch.max(output_map.data, 1)
                        num_corrects_ms += torch.sum(predicted_ms == map_labels).data.item()
            val_accuracy_rgb = (num_corrects_rgb / val_samples)
            avg_val_loss_rgb = val_loss_epoch_rgb / val_iter
            avg_val_loss_ms = val_loss_epoch_ms / val_iter
            avg_val_loss_ordering = val_loss_epoch_ordering / iterPerEpoch

            if config.ms_task == "classifier":
                val_accuracy_ms = (num_corrects_ms / map_pixel_samples)
            print('*****  Val: Epoch = {} | Loss {} | Accuracy = {} *****'.format(epoch + 1, avg_val_loss_rgb, val_accuracy_rgb))

            avg_val_loss_normalized_rgb = avg_val_loss_rgb if avg_val_loss_rgb < max_loss else max_loss
            avg_val_loss_normalized_ms = avg_val_loss_ms if avg_val_loss_ms < max_loss else max_loss
            if config.ms_task == "classifier":
                wandb.log({"valid_loss_rgb": avg_val_loss_normalized_rgb,
                           "valid_loss_ms": avg_val_loss_normalized_ms,
                           "valid_accuracy_rgb": val_accuracy_rgb,
                           "valid_accuracy_ms": val_accuracy_ms,
                           "avg_val_loss_ordering": avg_val_loss_ordering,
                           "eopch": (epoch + 1)})
            else:
                wandb.log({"valid_loss_rgb": avg_val_loss_normalized_rgb,
                           "valid_loss_ms": avg_val_loss_normalized_ms,
                           "valid_accuracy_rgb": val_accuracy_rgb,
                           "avg_val_loss_ordering": avg_val_loss_ordering,
                           "eopch": (epoch + 1)})


            if val_accuracy_rgb > best_accuracy:
                save_path_model = (config.models_dir + '/best_model_ms_state_dict.pth')
                torch.save(model.state_dict(), save_path_model)
                best_accuracy = val_accuracy_rgb
                wandb.run.summary["best_valid_accuracy"] = best_accuracy
        else:
            if (epoch + 1) % 10 == 0:
                save_path_model = (config.models_dir + '/best_model_ms_state_dict' + str(epoch + 1) + '.pth')
                # torch.save(model.state_dict(), save_path_model)
    wandb.run.summary["best_valid_accuracy"] = best_accuracy
    return
# %% PREPARE DATASET
from gtea_dataset import gtea61
from spatial_transforms import *
from objectAttentionModelConvLSTM import *

config = config_stage2_ordering
normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_rgb_list = [Scale(256), RandomHorizontalFlip(), MultiScaleCornerCrop([1, 0.875, 0.75, 0.65625], 224), ToTensor(), normalize]
if config.ms_task == 'classifier':
    transform_ms_list = transform_rgb_list[:3] + [Scale(7), ToTensor(), ToBinaryMap(config.binary_mask_threshold)]
else:
    transform_ms_list = transform_rgb_list[:3] + [Scale(7), ToTensor()]
transform_rgb = Compose(transform_rgb_list)
transform_ms = Compose(transform_ms_list)

gtea_root = "GTEA61"
train_dataset = gtea61("ms", gtea_root, split="train", user_split=config.training_user_split, seq_len_rgb=config.seq_len, transform_rgb=transform_rgb, transform_ms=transform_ms, preload=False)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)

val_transform = Compose([Scale(256), CenterCrop(224), ToTensor(), normalize])
if config.ms_task == 'classifier':
    val_transform_ms = Compose([Scale(256), CenterCrop(224), Scale(7), ToTensor(), ToBinaryMap(config.binary_mask_threshold)])
else:
    val_transform_ms = Compose([Scale(256), CenterCrop(224), Scale(7), ToTensor()])
val_dataset = gtea61("ms", gtea_root, split="test", user_split=config.val_user_split, seq_len_rgb=config.seq_len, transform_rgb=val_transform, transform_ms=val_transform_ms, preload=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, pin_memory=True)


# %% TRAINING STAGE 2 MS
config = config_stage2_ordering

model, train_params_rgb, train_params_ms, train_params_ordering = prepare_training_ms(config)
model.lstm_cell.train(True)
model.classifier.train(True)

model.to(config.device)

loss_fn_rgb = nn.CrossEntropyLoss()
if config.ms_task == "classifier":
    loss_fn_ms = nn.CrossEntropyLoss()
    # loss_fn_ms = FocalLoss(alpha=1, gamma=5)
else:
    loss_fn_ms = nn.MSELoss()
loss_fn_ordering = nn.CrossEntropyLoss()

optimizer_fn = torch.optim.Adam([{'params': train_params_rgb}, {'params': train_params_ms, 'lr': config.ms_lr}, {'params': train_params_ordering, 'lr': config.ms_lr}], lr=config.lr, weight_decay=config.weight_decay, eps=1e-4)
optim_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_fn, milestones=config.decay_steps, gamma=config.decay_factor)


training_time = datetime.now().strftime("%d-%b_%H-%M")
wandb.init(config=config, group=f"{config.seq_len}f", name=f"{training_time} Ordering, {config.ms_task}, {config.ms_features}, {config.seq_len}f", project="mldl-fpar")

training_ms(model, config, train_loader, val_loader)


#%% LR FINDER
# lr_find_epochs = 2
# start_lr = 1e-7
# end_lr = 1e-3
#
# lr_lambda = lambda x: math.exp(x * math.log(end_lr / start_lr) / (lr_find_epochs * len(train_loader)))
#
# config = config_stage2_ms
#
# model, train_params_rgb, train_params_ms = prepare_training_ms(config)
# model.lstm_cell.train(True)
# model.classifier.train(True)
#
# model.to(config.device)
#
# loss_fn_rgb = nn.CrossEntropyLoss()
# if config.ms_task == "classifier":
#     # loss_fn_ms = nn.CrossEntropyLoss()
#     loss_fn_ms = FocalLoss(alpha=1, gamma=5)
# else:
#     loss_fn_ms = nn.MSELoss()
#
# optimizer_fn = torch.optim.Adam([{'params': train_params_rgb}, {'params': train_params_ms, 'lr': start_lr}], lr=start_lr, weight_decay=config.weight_decay, eps=1e-4)
# optim_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_fn, lr_lambda)
# lr_find_loss_rgb = []
# lr_find_lr_rgb = []
# lr_find_loss_ms = []
# lr_find_lr_ms = []
# val_lr_find_loss_rgb = []
# val_lr_find_lr_rgb = []
# val_lr_find_loss_ms = []
# val_lr_find_lr_ms = []
# smoothing = 0.05
#
# # training_time = datetime.now().strftime("%d-%b_%H-%M")
# # wandb.init(config=config, group=f"{config.seq_len}f", name=f"{training_time} MS, {config.ms_task}, {config.ms_features}, {config.seq_len}f, FLoss", project="mldl-fpar")
#
# training_ms_lr_finder(model, config, train_loader, val_loader)
#
# #%%
# def training_ms_lr_finder(model, config, train_loader, val_loader):
#     train_iter = 0
#     best_accuracy = 0
#     for epoch in range(lr_find_epochs):
#         epoch_loss_rgb = 0
#         epoch_loss_ms = 0
#         num_corrects_rgb = 0
#         num_corrects_ms = 0
#         trainSamples = 0
#         map_pixel_samples = 0
#         iterPerEpoch = 0
#         model.lstm_cell.train(True)
#         model.classifier.train(True)
#         model.resNet.layer4[0].conv1.train(True)
#         model.resNet.layer4[0].conv2.train(True)
#         model.resNet.layer4[1].conv1.train(True)
#         model.resNet.layer4[1].conv2.train(True)
#         model.resNet.layer4[2].conv1.train(True)
#         model.resNet.layer4[2].conv2.train(True)
#         model.resNet.fc.train(True)
#         model.ms_conv.train(True)
#         model.ms_classifier.train(True)
#         # display_ms = True
#         for inputs_rgb, map_labels, labels in train_loader:
#             num_samples = inputs_rgb.size(0)
#             trainSamples += num_samples
#
#             iterPerEpoch += 1
#             optimizer_fn.zero_grad()
#             inputs_rgb = inputs_rgb.permute(1, 0, 2, 3, 4).to(config.device)  # but why?
#             labels = labels.to(config.device)
#             # map_labels = map_labels.to(config.device)
#             map_labels = map_labels.to(config.device).permute(0, 2, 1, 3, 4).squeeze() # BSxseq_lenx7x7
#             output_label, _, output_map = model(inputs_rgb)  # output_map is BSx2
#             # if display_ms:
#             #     display_map_prediction(map_labels[0].clone(), output_map[0].clone(), loss_fn_ms, config.ms_task)
#             #     display_ms = False
#             map_pixel_samples += num_samples * config.seq_len * 7 * 7
#             loss_rgb = loss_fn_rgb(output_label, labels)
#             loss_ms = loss_fn_ms(output_map.squeeze(), map_labels).mean()
#             loss = loss_rgb + loss_ms
#             loss.backward()
#             optimizer_fn.step()
#
#             optim_scheduler.step()
#             lr_step_rgb = optimizer_fn.state_dict()["param_groups"][0]["lr"]
#             lr_step_ms = optimizer_fn.state_dict()["param_groups"][0]["lr"]
#             lr_find_lr_rgb.append(lr_step_rgb)
#             lr_find_lr_ms.append(lr_step_ms)
#
#             if train_iter == 0:
#                 lr_find_loss_rgb.append(loss_rgb.item())
#                 lr_find_loss_ms.append(loss_ms.item())
#             else:
#                 loss_rgb_val = smoothing * loss_rgb.item() + (1 - smoothing) * lr_find_loss_rgb[-1]
#                 lr_find_loss_rgb.append(loss_rgb_val)
#                 loss_ms_val = smoothing * loss_ms.item() + (1 - smoothing) * lr_find_loss_ms[-1]
#                 lr_find_loss_ms.append(loss_ms_val)
#
#             train_iter += 1
#
#         # with torch.no_grad():
#         #     model.eval()
#         #     val_loss_epoch_rgb = 0
#         #     val_loss_epoch_ms = 0
#         #     val_iter = 0
#         #     val_samples = 0
#         #     num_corrects_rgb = 0
#         #     num_corrects_ms = 0
#         #     map_pixel_samples = 0
#         #     for inputs_rgb, map_labels, labels in val_loader:
#         #
#         #         num_samples = inputs_rgb.size(0)
#         #         val_samples += num_samples
#         #         inputs_rgb = inputs_rgb.permute(1, 0, 2, 3, 4).to(config.device)
#         #         labels = labels.to(config.device)
#         #         # map_labels = map_labels.to(config.device)
#         #         map_labels = map_labels.to(config.device).view(num_samples, config.seq_len, 7, 7)
#         #         output_label, _, output_map = model(inputs_rgb)
#         #         # map_labels = map_labels.view(num_samples * config.seq_len * 49)
#         #         # output_map = output_map.view(config.seq_len * num_samples * 49, 2)
#         #         map_pixel_samples += num_samples * config.seq_len * 7 * 7
#         #         val_loss_rgb = loss_fn_rgb(output_label, labels)
#         #         val_loss_ms = loss_fn_ms(output_map.squeeze(), map_labels)
#         #         val_loss_epoch_rgb += val_loss_rgb.item()
#         #         val_loss_epoch_ms += val_loss_ms.mean().item()
#         #
#         #         if val_iter == 0:
#         #             val_lr_find_loss_rgb.append(val_loss_rgb.item())
#         #             val_lr_find_loss_ms.append(val_loss_ms.item())
#         #         else:
#         #             val_loss_rgb_val = smoothing * val_loss_rgb.item() + (1 - smoothing) * val_lr_find_loss_rgb[-1]
#         #             val_lr_find_loss_rgb.append(val_loss_rgb_val)
#         #             val_loss_ms_val = smoothing * val_loss_ms.item() + (1 - smoothing) * val_lr_find_loss_ms[-1]
#         #             val_lr_find_loss_ms.append(val_loss_ms_val)
#         #
#         #         val_iter += 1
#
#
#     return