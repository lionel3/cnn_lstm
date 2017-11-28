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
import argparse

parser = argparse.ArgumentParser(description='cnn_lstm testing')
parser.add_argument('-g', '--gpu', default=[1], nargs='+', type=int, help='index of gpu to use, default 1')
parser.add_argument('-s', '--seq', default=4, type=int, help='sequence length, default 4')
parser.add_argument('-t', '--test', default=800, type=int, help='test batch size, default 800')
parser.add_argument('-w', '--work', default=2, type=int, help='num of workers to use, default 2')

args = parser.parse_args()
gpu_usg = ",".join(list(map(str, args.gpu)))
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg
sequence_length = args.seq
test_batch_size = args.test

use_10_crop = False
lstm_in_dim = 2048
lstm_out_dim = 512

num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()

from torchvision.transforms import Lambda

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


class resnet_lstm(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm, self).__init__()
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
        self.lstm = nn.LSTM(lstm_in_dim, lstm_out_dim, batch_first=True)
        self.fc = nn.Linear(lstm_out_dim, 7)
        self.fc2 = nn.Linear(2048, 7)
        init.xavier_normal(self.lstm.all_weights[0][0])
        init.xavier_normal(self.lstm.all_weights[0][1])

    def forward(self, x):
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        z = self.fc2(x)
        x = x.view(-1, sequence_length, lstm_in_dim)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, lstm_out_dim)
        y = self.fc(y)
        return z, y


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

    print(np.max(train_labels))
    print(np.max(val_labels))
    print(np.max(test_labels))

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
    if use_10_crop:

        test_transforms = transforms.Compose([
            transforms.TenCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack([transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(crop) for crop in crops]))
        ])
    else:
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
    print('num of test:', num_test)
    test_count = 0
    for i in range(len(test_num_each)):
        test_count += test_num_each[i]
    print('vertify num of test:', test_count)

    test_useful_start_idx = get_useful_start_idx(sequence_length, test_num_each)

    print('num of useful test start idx:', len(test_useful_start_idx))
    print('the last idx of test start idx:', test_useful_start_idx[-1])

    num_test_we_use = len(test_useful_start_idx)
    # num_test_we_use = 804

    test_we_use_start_idx = test_useful_start_idx[0:num_test_we_use]

    test_idx = []
    for i in range(num_test_we_use):
        for j in range(sequence_length):
            test_idx.append(test_we_use_start_idx[i] + j)

    num_test_all = len(test_idx)
    print('num of testset:', num_test)
    print('num of test samples we use:', num_test_we_use)
    print('num of all test samples:', num_test_all)
    print('test batch size:', test_batch_size)
    print('sequence length:', sequence_length)
    print('num of gpu:', num_gpu)

    # test_sampler = torch.utils.data.sampler.SequentialSampler(test_idx)
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        sampler=test_idx,
        # shuffle=True,
        num_workers=1,
        pin_memory=False
    )
    model = torch.load('cnn_lstm_epoch_25_length_4_opt_1_batch_200_train1_9951_train2_9800_val1_9680_val2_8468.pth')

    if use_gpu:
        model = model.cuda()
    # model = DataParallel(model)
    model = model.module
    criterion_1 = nn.BCEWithLogitsLoss(size_average=False)
    criterion_2 = nn.CrossEntropyLoss(size_average=False)
    sig_f = nn.Sigmoid()

    model.eval()
    test_loss_1 = 0.0
    test_loss_2 = 0.0
    test_corrects_1 = 0
    test_corrects_2 = 0

    test_start_time = time.time()
    for data in test_loader:
        inputs, labels_1, labels_2 = data
        if use_gpu:
            inputs = Variable(inputs.cuda(), volatile=True)
            labels_1 = Variable(labels_1.cuda(), volatile=True)
            labels_2 = Variable(labels_2.cuda(), volatile=True)
        else:
            inputs = Variable(inputs, volatile=True)
            labels_1 = Variable(labels_1, volatile=True)
            labels_2 = Variable(labels_2, volatile=True)

        outputs_1, outputs_2 = model.forward(inputs)

        _, preds_2 = torch.max(outputs_2.data, 1)

        sig_out = outputs_1.data.cpu()
        sig_out = sig_f(sig_out)
        preds_1 = torch.ByteTensor(sig_out > 0.5)
        preds_1 = preds_1.long()
        test_corrects_1 += torch.sum(preds_1 == labels_1.data.cpu())
        labels_1 = Variable(labels_1.data.float())

        loss_1 = criterion_1(outputs_1, labels_1)
        loss_2 = criterion_2(outputs_2, labels_2)

        test_loss_1 += loss_1.data[0]
        test_loss_2 += loss_2.data[0]
        test_corrects_2 += torch.sum(preds_2 == labels_2.data)

    test_elapsed_time = time.time() - test_start_time
    test_accuracy_1 = test_corrects_1 / num_test_all / 7
    test_accuracy_2 = test_corrects_2 / num_test_all
    test_average_loss_1 = test_loss_1 / num_test_all
    test_average_loss_2 = test_loss_2 / num_test_all

    # print(type(all_preds))
    # print(len(all_preds))
    # with open('', 'wb') as f:
    #     pickle.dump(all_preds, f)

    print('test completed in: {:2.0f}m{:2.0f}s test loss_1: {:4.4f} test loss_2: {:4.4f} test accu_1: {:.4f} test accu_2: {:.4f}'
          .format(test_elapsed_time // 60, test_elapsed_time % 60, test_average_loss_1, test_average_loss_2, test_accuracy_1, test_accuracy_2))

print()


def main():
    _, _, _, _, test_dataset, test_num_each = get_data('train_val_test_paths_labels.pkl')

    test_model(test_dataset, test_num_each)


if __name__ == "__main__":
    main()

print('Done')
print()
