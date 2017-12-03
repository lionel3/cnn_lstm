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
import random

parser = argparse.ArgumentParser(description='cnn_lstm testing')
parser.add_argument('-g', '--gpu', default=[1], nargs='+', type=int, help='index of gpu to use, default 1')
parser.add_argument('-s', '--seq', default=4, type=int, help='sequence length, default 4')
parser.add_argument('-t', '--test', default=80, type=int, help='test batch size, default 80')
parser.add_argument('-w', '--work', default=2, type=int, help='num of workers to use, default 2')
parser.add_argument('-a', '--average', default=False, type=bool, help='whether to use 10 crop, default False')

args = parser.parse_args()
gpu_usg = ",".join(list(map(str, args.gpu)))
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg
sequence_length = args.seq
test_batch_size = args.test

use_10_crop = args.average
print(use_10_crop)
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

    print('train_paths : {:6d} val_paths : {:6d} test_paths : {:6d} '.format(len(train_paths), len(val_paths), len(test_paths)))
    print('train_labels: {:6d} val_labels: {:6d} test_labels: {:6d}'.format(len(train_labels), len(val_labels), len(test_labels)))
    print('train_each  : {:6d} val_each  : {:6d} test_each  : {:6d}'.format(len(train_num_each), len(val_num_each), len(test_num_each)))

    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    # print(np.max(train_labels))
    # print(np.max(val_labels))
    # print(np.max(test_labels))

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
                lambda crops: torch.stack(
                    [transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(crop) for crop in crops]))
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
    test_count = 0
    for i in range(len(test_num_each)):
        test_count += test_num_each[i]

    test_useful_start_idx = get_useful_start_idx(sequence_length, test_num_each)
    print('num of test : {:6d} vertify num: {:6d} num_useful: {:6d} last_index: {:6d}'.format(num_test, test_count,
                                                                                              len(test_useful_start_idx),
                                                                                              test_useful_start_idx[-1]))
    num_test_we_use = len(test_useful_start_idx)
    # num_test_we_use = 804

    test_we_use_start_idx = test_useful_start_idx[0:num_test_we_use]

    test_idx = []
    for i in range(num_test_we_use):
        for j in range(sequence_length):
            test_idx.append(test_we_use_start_idx[i] + j)

    num_test_all = len(test_idx)
    print('num_test: {:6d} num_test_we_use: {:6d} num_test_all: {:6d}'.format(num_test, num_test_we_use, num_test_all))
    print(
        'num_gpu : {:6d} sequence_length: {:6d} test_batch  : {:6d}'.format(num_gpu, sequence_length, test_batch_size))

    # test_sampler = torch.utils.data.sampler.SequentialSampler(test_idx)
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        sampler=test_idx,
        # shuffle=True,
        num_workers=1,
        pin_memory=False
    )
    model = torch.load('cnn_lstm_epoch_25_length_10_opt_1_batch_400_train1_9993_train2_9971_val1_9692_val2_8647.pth')

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
    all_preds_1 = []
    all_labels_1 = []
    all_preds_2 = []

    for data in test_loader:
        inputs, labels_1, labels_2 = data

        if use_10_crop:
            # labels_1 = labels_1[(sequence_length - 1)::sequence_length]
            # shuffle_idx = []
            # for i in range(inputs.size()[0]):
            #     temp_idx = [i for i in range(10 * i, 10 * i + 10)]
            #     random.shuffle(temp_idx)
            #     shuffle_idx.extend(temp_idx)
            # print(shuffle_idx[0])
            # print(shuffle_idx[1])
            # x = shuffle_idx[0]
            # y = shuffle_idx[1]
            # print(inputs.size())
            # inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
            # print(inputs.size())
            # inputs = inputs.view(-1, 3, 224, 224)
            # print(inputs[x, 0, 0, 0])
            # print(inputs[y, 0, 0, 0])
            # inputs = inputs[np.asarray(shuffle_idx, dtype=int)]
            # print(inputs[0, 0, 0, 0])
            # print(inputs[1, 0, 0, 0])
            # print(inputs.size())
            labels_2 = labels_2[(sequence_length - 1)::sequence_length]

            for i in range(10):
                if use_gpu:
                    inputs_temp = Variable(inputs[:, i, :, :, :].cuda(), volatile=True)
                else:
                    inputs_temp = Variable(inputs, volatile=True)
                outputs_1_temp, outputs_2_temp = model.forward(inputs_temp)
                if i == 0:
                    outputs_1 = outputs_1_temp
                    outputs_2 = outputs_2_temp
                else:
                    outputs_1 = torch.cat((outputs_1, outputs_1_temp), 0)
                    outputs_2 = torch.cat((outputs_2, outputs_2_temp), 0)
            # print(outputs_1.size())
            outputs_1 = outputs_1.view(10, -1, 7)
            outputs_2 = outputs_2.view(10, -1, 7)
            # print(outputs_1.size())
            outputs_1 = torch.mean(outputs_1, 0)
            outputs_2 = torch.mean(outputs_2, 0)
            # print(outputs_1.size())
            # print(outputs_2.size())
            if use_gpu:
                labels_1 = Variable(labels_1.cuda(), volatile=True)
                labels_2 = Variable(labels_2.cuda(), volatile=True)
            else:
                labels_1 = Variable(labels_1, volatile=True)
                labels_2 = Variable(labels_2, volatile=True)

            outputs_2 = outputs_2[sequence_length - 1::sequence_length]
            _, preds_2 = torch.max(outputs_2.data, 1)

            for i in range(len(outputs_1)):
                all_preds_1.append(outputs_1[i].data.cpu().numpy().tolist())
                all_labels_1.append(labels_1[i].data.cpu().numpy().tolist())
            for i in range(len(preds_2)):
                all_preds_2.append(preds_2[i])
            print('preds_1: {:6d} preds_2: {:6d}'.format(len(all_preds_1), len(all_preds_2)))

            labels_1 = Variable(labels_1.data.float())
            loss_1 = criterion_1(outputs_1, labels_1)
            loss_2 = criterion_2(outputs_2, labels_2)

            test_loss_1 += loss_1.data[0]
            test_loss_2 += loss_2.data[0]
            test_corrects_2 += torch.sum(preds_2 == labels_2.data)

        else:

            # labels_1 = labels_1[(sequence_length - 1)::sequence_length]
            labels_2 = labels_2[(sequence_length - 1)::sequence_length]
            if use_gpu:
                inputs = Variable(inputs.cuda(), volatile=True)
                labels_1 = Variable(labels_1.cuda(), volatile=True)
                labels_2 = Variable(labels_2.cuda(), volatile=True)
            else:
                inputs = Variable(inputs, volatile=True)
                labels_1 = Variable(labels_1, volatile=True)
                labels_2 = Variable(labels_2, volatile=True)

            outputs_1, outputs_2 = model.forward(inputs)

            # outputs_1 = outputs_1[sequence_length-1::sequence_length]
            outputs_2 = outputs_2[sequence_length - 1::sequence_length]

            _, preds_2 = torch.max(outputs_2.data, 1)

            for i in range(len(outputs_1)):
                all_preds_1.append(outputs_1[i].data.cpu().numpy().tolist())
                all_labels_1.append(labels_1[i].data.cpu().numpy().tolist())
            for i in range(len(preds_2)):
                all_preds_2.append(preds_2[i])
            print('preds_1: {:6d} preds_2: {:6d}'.format(len(all_preds_1), len(all_preds_2)))

            labels_1 = Variable(labels_1.data.float())
            loss_1 = criterion_1(outputs_1, labels_1)
            loss_2 = criterion_2(outputs_2, labels_2)

            test_loss_1 += loss_1.data[0]
            test_loss_2 += loss_2.data[0]
            test_corrects_2 += torch.sum(preds_2 == labels_2.data)

    all_preds_1_cor = []
    all_labels_1_cor = []
    cor_count = 0
    for i in range(len(test_num_each)):
        for j in range(cor_count, cor_count + test_num_each[i] - (sequence_length - 1)):
            if j == cor_count:
                for k in range(sequence_length - 1):
                    all_preds_1_cor.append(all_preds_1[sequence_length * j + k])
                    all_labels_1_cor.append(all_labels_1[sequence_length * j + k])
            all_preds_1_cor.append(all_preds_1[sequence_length * j + sequence_length - 1])
            all_labels_1_cor.append(all_labels_1[sequence_length * j + sequence_length - 1])
        cor_count += test_num_each[i] + 1 - sequence_length

    print('all_preds_1 : {:6d}'.format(len(all_preds_1)))
    print('all_labels_1: {:6d}'.format(len(all_labels_1)))
    print('cor_labels_1: {:6d}'.format(len(all_preds_1_cor)))
    print('cor_labels_1: {:6d}'.format(len(all_labels_1_cor)))

    pt_preds_1 = torch.from_numpy(np.asarray(all_preds_1_cor, dtype=np.float32))
    pt_labels_1 = torch.from_numpy(np.asarray(all_labels_1_cor, dtype=np.float32))
    print('pt preds_1 :', pt_preds_1.size())
    print('pt labels_1:', pt_labels_1.size())
    sig_out = sig_f(pt_preds_1)
    preds_cor = torch.ByteTensor(sig_out > 0.5)
    preds_cor = preds_cor.long()
    pt_labels_1 = pt_labels_1.long()
    test_corrects_1 += torch.sum(preds_cor == pt_labels_1)

    test_elapsed_time = time.time() - test_start_time
    test_accuracy_1 = test_corrects_1 / (num_test_we_use + sequence_length - 1) / 7
    test_accuracy_2 = test_corrects_2 / num_test_we_use
    test_average_loss_1 = test_loss_1 / (num_test_we_use + sequence_length - 1) / 7
    test_average_loss_2 = test_loss_2 / num_test_we_use

    print('preds_1 num: {:6d} preds_2 num: {:6d}'.format(len(all_preds_1_cor), len(all_preds_2)))
    # with open('cnn_lstm_epoch_25_length_10_opt_1_batch_400_train1_9993_train2_9971_val1_9692_val2_8647_preds_10_1.pkl', 'wb') as f:
    #     pickle.dump(all_preds_1, f)
    # with open('cnn_lstm_epoch_25_length_10_opt_1_batch_400_train1_9993_train2_9971_val1_9692_val2_8647_preds_10_2.pkl', 'wb') as f:
    #     pickle.dump(all_preds_2, f)

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
