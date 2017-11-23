import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import time
from torch.nn import DataParallel
import os
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time

import pickle
import numpy as np

# print(torch.cuda.device_count())
num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class CholecDataset(Dataset):
    def __init__(self, file_paths, file_labels, transform=None,
                 loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels_1 = file_labels[:, range(7)]
        self.file_labels_2 = file_labels[:, -1]
        self.transform = transform
        # self.target_transform=target_transform
        self.loader = loader

    def __getitem__(self, index):
        img_names = self.file_paths[index]
        labels_1 = self.file_labels_1[index]
        labels_2 = self.file_labels_2[index]
        imgs = self.loader(img_names)
        if self.transform is not None:
            imgs = self.transform(imgs)

        return imgs, labels_1, labels_2

    def __len__(self):
        return len(self.file_paths)

test_batch_size = 800

class my_resnet(torch.nn.Module):
    def __init__(self):
        super(my_resnet, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)
        # self.task_1 = nn.Sequential()
        # self.task_1.add_module("lstm", nn.LSTM(lstm_in_dim, 7))
        # self.fc = nn.Linear(2048, lstm_in_dim)
        self.fc = nn.Linear(2048, 7)

    def forward(self, x):
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        x = self.fc(x)

        return x


def get_useful_start_idx(sequence_length, list_each_length):
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)
        count += list_each_length[i]
    return idx


def get_data(data_path):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    train_paths = train_test_paths_labels[0]
    val_paths = train_test_paths_labels[1]
    test_paths = train_test_paths_labels[2]
    train_labels = train_test_paths_labels[3]
    val_labels = train_test_paths_labels[4]
    test_labels = train_test_paths_labels[5]
    train_num_each = train_test_paths_labels[6]
    val_num_each = train_test_paths_labels[7]
    test_num_each = train_test_paths_labels[8]

    print('train_paths:', len(train_paths))
    print('val_paths:', len(val_paths))
    print('test_paths:', len(test_paths))
    print('train_labels:', len(train_labels))
    print('val_labels:', len(val_labels))
    print('test_labels:', len(test_labels))

    print('train_num_each:', len(train_num_each))
    print('val_num_each:', len(val_num_each))
    print('test_num_each:', len(test_num_each))

    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    # print(test_labels[0].shape)
    # print(val_labels[0].shape)
    # print(test_labels[0].shape)

    train_transforms = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    val_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    test_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    train_dataset = CholecDataset(train_paths, train_labels, train_transforms)
    val_dataset = CholecDataset(val_paths, val_labels, val_transforms)
    test_dataset = CholecDataset(test_paths, test_labels, test_transforms)

    return train_dataset, train_num_each, val_dataset, val_num_each, test_dataset, test_num_each


def test_model(test_dataset, test_num_each):
    num_test = len(test_dataset)
    # num_test = 800
    # test_idx = [i for i in range(1000)]
    test_idx = [i for i in range(num_test)]

    num_test_all = 7 * num_test

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        sampler=test_idx,
        # shuffle=True,
        num_workers=8,
        pin_memory=False
    )
    model = torch.load('20171121_epoch_25_multi_cnn_adam.pth')
    criterion = nn.BCEWithLogitsLoss(size_average=False)
    model = model.module
    sig_f = nn.Sigmoid()
    model.eval()
    test_loss = 0.0
    test_corrects = 0
    test_start_time = time.time()
    all_preds = []
    for data in test_loader:
        inputs, labels_1, labels_2 = data
        if use_gpu:
            inputs = Variable(inputs.cuda(),volatile=True)
            labels = Variable(labels_1.cuda(),volatile=True)
        else:
            inputs = Variable(inputs,volatile=True)
            labels = Variable(labels_1,volatile=True)

        outputs = model.forward(inputs)

        for i in range(len(outputs)):
            all_preds.append(outputs[i].data.cpu().numpy().tolist())
        print(len(all_preds))
        # print(all_preds[0])
        sig_out = outputs.data.cpu()
        sig_out = sig_f(sig_out)

        predict = torch.ByteTensor(sig_out > 0.5)
        predict = predict.long()
        test_corrects += torch.sum(predict == labels.data.cpu())
        # print(test_corrects)
        labels = Variable(labels.data.float())
        loss = criterion(outputs, labels)
        test_loss += loss.data[0]
        # print(test_corrects)
    test_elapsed_time = time.time() - test_start_time
    test_accuracy = test_corrects / num_test_all
    test_average_loss = test_loss / num_test_all
    print(type(all_preds))
    print('all preds:', len(all_preds))
    # print(all_preds[0])
    # all_preds_np = np.asarray(all_preds, dtype=np.float32)
    with open('20171121_epoch_25_multi_cnn_adam_preds.pkl', 'wb') as f:
        pickle.dump(all_preds, f)
    # np.save('20171121_epoch_25_multi_cnn_adam_preds.pkl', all_preds_np)
    print('test completed in: {:2.0f}m{:2.0f}s  test loss: {:4.4f} test accu: {:.4f}'
          .format(test_elapsed_time // 60, test_elapsed_time % 60, test_average_loss, test_accuracy))


print()


def main():
    _, _, _, _, test_dataset, test_num_each = get_data('train_val_test_paths_labels.pkl')
    test_model(test_dataset, test_num_each)

    # _, _, _, _, test_dataset, test_num_each = get_data('train_val_test_paths_labels.pkl')
    # test_model(test_dataset)


if __name__ == "__main__":
    main()

print('Done')
print()
