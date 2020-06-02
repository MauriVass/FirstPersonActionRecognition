import torch
import tqdm
import os
import shutil

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