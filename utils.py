import torch
import tqdm
import os
import shutil
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage


class Config(object):

    def __init__(self, config_dict=None, device='cuda'):
        self.__setattr__("device", device)
        if config_dict is not None:
            for name, value in config_dict.items():
                self.__setattr__(name, value)

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def as_dict(self):
        return {key: val for key, val in self.__dict__.items()}

    def __str__(self):
        return str(self.__dict__)

    def update(self, config):
        self.__dict__.update(config.as_dict())


def test(net, test_dataset, test_dataloader, config, compute_loss=True, loss_criterion=None, progress=False, printout=False):
    net = net.to(config.device)
    net.eval()
    running_loss = 0
    running_corrects = 0
    num_batches = len(test_dataloader)
    for images, labels in (tqdm(test_dataloader) if progress else test_dataloader):
        images = images.to(config.device)
        labels = labels.to(config.device)
        outputs = net(images)  # forward pass
        _, preds = torch.max(outputs.data, 1)  # get predictions
        running_corrects += torch.sum(preds == labels.data).data.item()  # update corrects
        if compute_loss:
            running_loss += loss_criterion(outputs, labels).item()

    accuracy = running_corrects / float(len(test_dataset))
    loss = running_loss / num_batches
    if printout:
        print(f"Test: accuracy = {accuracy}", f", loss ={loss}" if compute_loss else "")

    return (accuracy, loss) if compute_loss else accuracy


def store_model(filename, run_id):
    dest_path = os.path.join("models_container", run_id)
    source_path = os.path.join("models", filename)

    os.makedirs(dest_path, exist_ok=True)
    shutil.copy(source_path, dest_path)


def display_map_prediction(ground_truth, prediction, loss_criterion):
    # map_labels is seq_len x7x7
    # prediction is seq_len x2xx7x7
    ground_truth = ground_truth.squeeze().cpu()
    prediction = prediction.squeeze().cpu().permute(1, 0, 2, 3) # .to(torch.uint8)
    topil = ToPILImage()
    seq_len = ground_truth.size(0)
    rows = 2
    fig = plt.figure(figsize=(14, 4))
    tot_loss = 0
    tot_corrects = 0
    for i in range(seq_len):
        ground_truth_img = topil(ground_truth[i].to(torch.uint8)).convert("L")
        ax = fig.add_subplot(rows, seq_len, i+1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.imshow(ground_truth_img) # prediction[0].unsqueeze(0)
        loss = loss_criterion(prediction[i].unsqueeze(0), ground_truth[i].to(torch.int64).unsqueeze(0))
        _, predicted_map = torch.max(prediction[i].data, 0)
        num_corrects = torch.sum(predicted_map == ground_truth[i]).data.item()
        predicted_img = topil(predicted_map.to(torch.uint8)).convert("L")
        ax = fig.add_subplot(rows, seq_len, i+1 + seq_len)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        accuracy = num_corrects/(7*7)
        ax.set_title(f"A:{accuracy:.1f}|L={loss.item():.1f}")
        plt.imshow(predicted_img)
        # topil = ToPILImage()
        # img = topil(t_map).convert("RGB")
        # plt.imshow(img)
    plt.show()

    pass