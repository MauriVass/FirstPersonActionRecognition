from torchvision.datasets import VisionDataset
from spatial_transforms import Normalize
from PIL import Image
from math import ceil
import numpy as np
import random
import os
import sys
import torch
import time
import pandas as pd

# directory containing the x-flows frames
FLOW_X_DIR = "flow_x_processed"
# directory containing the y-flows frames
FLOW_Y_DIR = "flow_y_processed"
# directory containing the rgb frames
RGB_DIR = "processed_frames2"


""""
    Datasets are to be built by calling gtea61() and passing, among other arguments,
    the type of dataset to build.
    Allowed types are:
        rgb:    rgb frames
        flow:   warp-flow frames
        ms:     rgb + motion-segmentation frames
        joint:  rgb + warp-flow frames for the joint training
    Note that if your RAM allows it, you can pass preload=True to preload the frames and
    speed up item retrieval during training.
"""


def pil_loader(path, image_type):  # type is eiter RGB or L
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(image_type)

def entropy_based_frame_sampler(start, end, seq_len, path):
    random_value = random.random()

    if random_value >= 0 and random_value <= 0.33:

        path.replace("mmaps", "rgb")
        path.replace("map", "rgb")

        entropies = pd.read_csv("entropies/" + path + "/entropies.txt")
        entropies_sorted = entropies.sort_values(by=entropies.columns[4], ascending = False)[:seq_len]

        frames = [
          frame[1] for frame in entropies_sorted.values.tolist()
        ]

        if (len(frames) < seq_len):
          return uniform_frame_sampler(start, end, seq_len, path)

        return np.array(sorted(frames))

    elif random_value > 0.33 and random_value <= 0.66:
        return uniform_frame_sampler(start, end, seq_len, path)

    elif random_value > 0.66 and random_value <= 1:

        path.replace("mmaps", "rgb")
        path.replace("map", "rgb")

        entropies = pd.read_csv("entropies/" + path + "/entropies.txt")
        entropies_sorted = entropies.sort_values(by=entropies.columns[4], ascending = True)[:seq_len]

        frames = [
          frame[1] for frame in entropies_sorted.values.tolist()
        ]

        if (len(frames) < seq_len):
          return uniform_frame_sampler(start, end, seq_len, path)
        return np.array(sorted(frames))
    return


def uniform_frame_sampler(start, end, seq_len, path):
    return np.linspace(start, end, seq_len, endpoint=False, dtype=int)


def sequential_frame_sampler(start, end, seq_len, starting_seq, path, seed=None):
    # starting_frame mode is either first, center, or random
    if starting_seq == "first":
        return np.arange(start, seq_len)
    elif starting_seq == "center":
        starting_frame = np.ceil((end - seq_len) / 2)
        return int(starting_frame) + np.arange(start, seq_len, dtype=int)
    else:
        if seed is not None:
            random.seed(seed)
        starting_frame = random.randint(start, end - seq_len)
        return int(starting_frame) + np.arange(start, seq_len, dtype=int)


def allin_frame_sampler(start, end, seq_len):
    return list(range(start, end))


def gtea61(data_type, root, split='train', user_split=None, seq_len_rgb=7, seq_len_flow=5, transform_rgb=None, transform_flow=None, transform_ms=None, preload=False, *args, **kwargs):
    # type is rgb, flow, joint, ms
    if user_split is None:
        #  select users to source data from (out of [S1, S2, S3, S4])
        #  if no split is provided, it defaults to standard split
        if split == "train":
            user_split = [1, 3, 4]
        else:
            user_split = [2]

    if data_type == "rgb":
        return GTEA61_RGB(root, split, user_split, seq_len_rgb, preload, transform_rgb, *args, **kwargs)
    elif data_type == "flow":
        return GTEA61Flow(root, split, user_split, seq_len_flow, preload, transform_flow, *args, **kwargs)
    elif data_type == "joint":  # both rgb and flow
        return GTEA61_2Stream(root, split, user_split, seq_len_rgb, seq_len_flow, preload, transform_rgb, transform_flow, *args, **kwargs)
    elif data_type == "ms":  # both rgb and ms
        return GTEA61_MS(root, split, user_split, seq_len_rgb, preload, transform_rgb, transform_ms, *args, **kwargs)


class GTEA61(VisionDataset):
    def __init__(self, root, split, user_split, seq_len, preload, transform=None, target_transform=None, *args, **kwargs):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.root_dir = root
        self.split = split
        self.user_split = user_split  # selected users to source data from (out of [S1, S2, S3, S4])
        self.seq_len = seq_len
        self.label_names = []  # label_names[i] holds action name for class i
        self.labels = []    # labels[i] holds the class id of the video i
        self.preloaded = preload  # if true, images will be pre-loaded into memory
        self.transform = transform

    def build_metadata(self, data_main_path, paths_holder):
        #  builds and stores paths for each video instance
        #  also builds and stores label_names and class id mappings
        main_path = os.path.join(self.root_dir, data_main_path)
        for user_index in self.user_split:
            user_dir = f"S{user_index}"
            for action in sorted(os.listdir(os.path.join(main_path, user_dir))):
                if action not in self.label_names:
                    self.label_names.append(action)
                label_index = self.label_names.index(action)
                action_path = os.path.join(main_path, user_dir, action)
                for video_instance in sorted(os.listdir(action_path)):
                    self.labels.append(label_index)
                    paths_holder.append(os.path.join(action_path, video_instance))

    def load_frames(self, path, frame_sampler, image_type, *args):
        # loads the sequence of images for the video in path according to the frame_sampler
        frames = np.array(sorted(os.listdir(path)))
        frames_num = len(frames)
        sampled_frames = frames[frame_sampler(0, frames_num, self.seq_len, path, *args)]
        return [pil_loader(os.path.join(path, file_path), image_type) for file_path in sampled_frames]

    def __getitem__(self, index):
        raise NotImplementedError  # MUST override

    def __len__(self):
        raise NotImplementedError  # MUST override

class GTEA61_RGB(GTEA61):
    def __init__(self, root, split, user_split, seq_len, preload=False, transform=None, target_transform=None, frame_sampler=None):
        super().__init__(root, split, user_split, seq_len, preload, transform=transform, target_transform=target_transform)
        # frames are taken uniformly spaced by defult, pass a callback to overwrite the sampling method
        # such callback is to generate indices corresponding to the frames to be sampled
        if frame_sampler is None:
            self.frame_sampler = uniform_frame_sampler
        elif frame_sampler == "entropy_based":
            self.frame_sampler = entropy_based_frame_sampler

        self.video_paths = []  # holds a path for each video
        self.build_metadata(RGB_DIR, self.video_paths)

        if self.preloaded:
            self.loaded_frames = []  # holds preloaded sequences of images
            for video_instance_path in self.video_paths:
                frames_path = os.path.join(video_instance_path, "rgb")
                self.loaded_frames.append(self.load_frames(frames_path, self.frame_sampler, "RGB"))

    def __getitem__(self, index):
        if self.preloaded:
            frames = self.loaded_frames[index]
        else:
            frames = self.load_frames(os.path.join(self.video_paths[index], "rgb"), self.frame_sampler, "RGB")
        self.transform.randomize_parameters()
        frames = [self.transform(image) for image in frames]
        sequence = torch.stack(frames, 0)
        return sequence, self.labels[index]

    def __len__(self):
        return len(self.video_paths)


class GTEA61Flow(GTEA61):
    def __init__(self, root, split, user_split, seq_len, preload=False, transform=None, target_transform=None, frame_sampler=None):
        super().__init__(root, split, user_split, seq_len, preload, transform=transform, target_transform=target_transform)
        # frames are taken sequentially by defult, pass a callback to overwrite the sampling method
        # such callback is to generate indices corresponding to the frames to be sampled
        self.split = split
        if frame_sampler is None:
            self.frame_sampler = sequential_frame_sampler
        elif frame_sampler == "entropy_based":
            self.frame_sampler = entropy_based_frame_sampler

        if self.split == "train":
            self.starting_seq = "random"
        else:
            self.starting_seq = "center"
        self.video_x_paths = []  # holds a path for each x flow video
        self.build_metadata(FLOW_X_DIR, self.video_x_paths)
        self.video_y_paths = [path.replace("flow_x_processed", "flow_y_processed") for path in self.video_x_paths]

        if self.preloaded:
            self.loaded_x_frames = []
            self.loaded_y_frames = []
            for video_x_instance_path, video_y_instance_path in zip(self.video_x_paths, self.video_x_paths):
                if self.starting_seq == "random":  # loads in all frames, will be sampled when requested
                    x_frames = self.load_frames(video_x_instance_path, allin_frame_sampler, "L")
                    y_frames = self.load_frames(video_y_instance_path, allin_frame_sampler, "L")
                else:  # loads in only necessary frames
                    x_frames = self.load_frames(video_x_instance_path, self.frame_sampler, "L", self.starting_seq)
                    y_frames = self.load_frames(video_y_instance_path, self.frame_sampler, "L", self.starting_seq)
                self.loaded_x_frames.append(x_frames)
                self.loaded_y_frames.append(y_frames)

    def __getitem__(self, index):
        if self.preloaded:
            if self.starting_seq == "random":  # randomly sample the loaded frames
                num_frames = len(self.loaded_x_frames[index])
                sampled_frames = self.frame_sampler(0, num_frames, self.seq_len, "random")
                stacked_frames = [frames[i] for i in sampled_frames for frames in (self.loaded_x_frames[index], self.loaded_y_frames[index])]
            else:  # frames are loaded and sampled, just stack them
                x_frames, y_frames = self.loaded_x_frames[index], self.loaded_y_frames[index]
                stacked_frames = [frames[i] for i in range(len(x_frames)) for frames in (x_frames, y_frames)]
        else:  # sample and load the frames
            seed = time()
            x_path, y_path = self.video_x_paths[index], self.video_y_paths[index]
            x_frames = self.load_frames(x_path, self.frame_sampler, "L", self.starting_seq, seed)
            y_frames = self.load_frames(y_path, self.frame_sampler, "L", self.starting_seq, seed)
            stacked_frames = [frames[i] for i in range(len(x_frames)) for frames in (x_frames, y_frames)]
            # stacked_frames = [None] * (len(x_frames) + len(y_frames))
            # stacked_frames[::2] = x_frames
            # stacked_frames[1::2] = y_frames
        self.transform.randomize_parameters()
        # x frames are transformed differently from y frames
        frames = [self.transform(image, inv=True, flow=True) if i % 2 == 0 else self.transform(image, inv=False, flow=True) for i, image in enumerate(stacked_frames)]
        sequence = torch.stack(frames, 0).squeeze(1)
        return sequence, self.labels[index]

    def __len__(self):
        return len(self.video_x_paths)


class GTEA61_2Stream():
    def __init__(self, root, split, user_split, seq_len_rgb, seq_len_flow, preload=False, transform_rgb=None, transform_flow=None, target_transform=None, frame_sampler_rgb=None, frame_sampler_flow=None):
        self.rgb_dataset = GTEA61_RGB(root, split, user_split, seq_len_rgb, preload=preload, transform=transform_rgb, frame_sampler=frame_sampler_rgb)
        self.flow_dataset = GTEA61Flow(root, split, user_split, seq_len_flow, preload=preload, transform=transform_flow, frame_sampler=frame_sampler_flow)
        self.split = split
        self.root = root
        self.user_split = self.rgb_dataset.user_split
        self.user_split = self.rgb_dataset.user_split
        self.seq_len_rgb = self.rgb_dataset.seq_len
        self.seq_len_flow = self.flow_dataset.seq_len
        self.label_names = self.rgb_dataset.label_names
        self.labels = self.rgb_dataset.labels
        self.preloaded = self.rgb_dataset.preloaded
        self.transform_rgb = transform_rgb
        self.transform_flow = transform_flow


    def __getitem__(self, index):
        rgb_sequence, label = self.rgb_dataset.__getitem__(index)
        flow_sequence, _ = self.flow_dataset.__getitem__(index)
        return flow_sequence, rgb_sequence, label

    def __len__(self):
        return len(self.rgb_dataset)


class GTEA61_MS(GTEA61):
    def __init__(self, root, split, user_split, seq_len, preload=False, transform_rgb=None, transform_ms=None, target_transform=None, frame_sampler=None):
        super().__init__(root, split, user_split, seq_len, preload, transform=transform_rgb, target_transform=target_transform)
        # frames are taken uniformly spaced by defult, pass a callback to overwrite the sampling method
        # such callback is to generate indices corresponding to the frames to be sampled
        if frame_sampler is None:
            self.frame_sampler = uniform_frame_sampler
        elif frame_sampler == "entropy_based":
            self.frame_sampler = entropy_based_frame_sampler

        self.video_paths = []  # holds a path for each video
        self.build_metadata(RGB_DIR, self.video_paths)
        self.transform_ms = transform_ms
        if self.preloaded:
            self.loaded_frames = []  # holds preloaded sequences of images
            self.loaded_maps = []
            for video_instance_path in self.video_paths:
                frames_path = os.path.join(video_instance_path, "rgb")
                self.loaded_frames.append(self.load_frames(frames_path, self.frame_sampler, "RGB"))

                maps_path = os.path.join(video_instance_path, "mmaps")
                self.loaded_maps.append(self.load_frames(maps_path, self.frame_sampler, "L"))

    def __getitem__(self, index):
        if self.preloaded:
            frames = self.loaded_frames[index]
            maps = self.loaded_maps[index]
        else:
            frames = self.load_frames(os.path.join(self.video_paths[index], "rgb"), self.frame_sampler, "RGB")
            maps = self.load_frames(os.path.join(self.video_paths[index], "mmaps"), self.frame_sampler, "L")
        self.transform.randomize_parameters()
        frames = [self.transform(image) for image in frames]
        maps = [self.transform_ms(image) for image in maps]
        sequence = torch.stack(frames, 0)
        sequence_maps = torch.stack(maps, 0)
        return sequence, sequence_maps, self.labels[index]

    def __len__(self):
        return len(self.video_paths)
