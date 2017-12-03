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


# batch_size 要整除gpu个数 以及sequence长度
sequence_length = 4
test_batch_size = 80
lstm_in_dim = 2048
lstm_out_dim = 512


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
        self.lstm = nn.LSTM(lstm_in_dim, lstm_out_dim)

        self.hidden = self.init_hidden()
        # print(len(self.lstm.all_weights))
        # print(len(self.lstm.all_weights[0]))
        self.fc = nn.Linear(lstm_out_dim, 7)

        init.xavier_normal(self.lstm.all_weights[0][0])
        init.xavier_normal(self.lstm.all_weights[0][1])
        # init.xavier_normal(self.fc.parameters())
        # self.count = 0
        # 多GPU时候.这种赋值方式不成功, 所以尽量取能整除的batch
        # self.forward_batch_size = 0

    def init_hidden(self, hidden_batch_size=1):
        if use_gpu:
            return (Variable(torch.zeros(1, hidden_batch_size, lstm_out_dim).cuda()),
                    Variable(torch.zeros(1, hidden_batch_size, lstm_out_dim).cuda()))
        else:
            return (Variable(torch.zeros(1, hidden_batch_size, lstm_out_dim)),
                    Variable(torch.zeros(1, hidden_batch_size, lstm_out_dim)))

    def forward(self, x):
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        # x = self.fc(x)
        # x = x.view(-1, 100, 1, 1)
        # print('x', x.size())
        # self.count += x.size()[0]
        # self.forward_batch_size = x.size()[0]

        # self.hidden = self.init_hidden(train_batch_size // num_gpu)
        # print('count', self.count)
        # 这边会出现问题, 因为view是根据最后一个维度来的,所以顺序不对, permute或者batch_fisrt解决问题
        x = x.view(-1, sequence_length, lstm_in_dim)
        x = x.permute(1, 0, 2)
        self.lstm.flatten_parameters()
        y, self.hidden = self.lstm(x, self.hidden)
        # print('hidden:', self.hidden[0].size())
        # print(self.hidden[0][0, 28])
        # print('y:', y.size())
        # print(y[2,28])
        # y = y.contiguous().view(num_gpu, sequence_length, -1, 7)
        # y = y.permute(0, 2, 1, 3).contiguous()
        # # transpose或者permute会把变量变成非连续(内存)contiguous, 需要加contiguous()来搞定,
        # # 看来不是内存地址的错,是因为没有写进forward函数里面
        # 结果什么意思,我为什么遇到原来的错误??? 以后一定切记留下错误的代码作比对
        # 可能错怪地址连续问题了, 很可能是多GPU的错误??? 但是多gpu刚开始结果也是百分之三四十的, 不是百分之四五
        # y = y.view((train_batch_size, 7))
        y = y.contiguous().view(1, sequence_length, -1, lstm_out_dim)
        y = y.permute(0, 2, 1, 3)
        y = y.contiguous().view((-1, lstm_out_dim))
        # print(y.size())
        y = self.fc(y)
        return y



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

    test_transforms = transforms.Compose([
        transforms.TenCrop(224),
        Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        Lambda(lambda crops: torch.stack([transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(crop) for crop in crops]))
        # transforms.ToTensor(),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    train_dataset = CholecDataset(train_paths, train_labels, train_transforms)
    val_dataset = CholecDataset(val_paths, val_labels, val_transforms)
    test_dataset = CholecDataset(test_paths, test_labels, test_transforms)

    return train_dataset, train_num_each, val_dataset, val_num_each, test_dataset, test_num_each


# 是不是要normalize???

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
    # num_test_we_use = len(test_useful_start_idx) // (test_batch_size // sequence_length) * (
    #     test_batch_size // sequence_length)


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
    model = torch.load('20171122_lstm_epoch_25_length_4_sgd_valid.pth')

    if use_gpu:
        model = model.cuda()
    # model = DataParallel(model)
    model = model.module
    criterion = nn.CrossEntropyLoss(size_average=False)

    model.eval()

    test_loss = 0.0
    test_corrects = 0
    test_start_time = time.time()

    all_preds = []

    for data in test_loader:
        inputs, labels_1, labels_2 = data
        labels_2 = labels_2[3::4]
        if use_gpu:
            inputs = Variable(inputs.cuda(), volatile=True)
            labels = Variable(labels_2.cuda(), volatile=True)
        else:
            inputs = Variable(inputs, volatile=True)
            labels = Variable(labels_2, volatile=True)

        model.hidden = model.init_hidden(len(data[0]) // sequence_length // num_gpu * 10)
        # 如果不在内部调用, 会出现显存持续增长的问题, 还不知道为什么
        # print(inputs[0, 0, 0, 0, 0])
        # print(inputs[0, 1, 0, 0, 0])
        # print(inputs[0, 2, 0, 0, 0])
        # print(inputs.size())
        inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
        # print(inputs.size())
        inputs = inputs.view(-1, 3, 224, 224)
        # print(inputs[0, 0, 0, 0])
        # print(inputs[80, 0, 0, 0])
        # print(inputs[160, 0, 0, 0])
        outputs = model.forward(inputs)
        # print(outputs.size())
        # print(outputs[0, 0])
        # print(outputs[80, 0])
        outputs = outputs.view(10, -1, 7)
        # print(outputs[0, 0, 0])
        # print(outputs[1, 0, 0])
        # print(outputs.size())
        # sum = 0
        # for i in range(10):
        #     sum += outputs[i, 0, 0]
        # print(sum)
        outputs = torch.mean(outputs, 0)
        # print(outputs.size())
        # print(outputs[0,0])

        # print(labels.size())
        # print(outputs.size())
        outputs = outputs[3::4]
        # print(outputs.size())
        # print(labels.size())
        _, preds = torch.max(outputs.data, 1)
        for i in range(len(preds)):
            all_preds.append(preds[i])
        print(len(all_preds))
        loss = criterion(outputs, labels)
        test_loss += loss.data[0] / len(data[0]) * 4
        test_corrects += torch.sum(preds == labels.data)
        # print(test_corrects)
    test_elapsed_time = time.time() - test_start_time
    test_accuracy = test_corrects / num_test_we_use
    test_average_loss = test_loss / num_test_we_use

    print(type(all_preds))
    print(len(all_preds))
    with open('20171122_lstm_epoch_25_length_4_sgd_preds_10.pkl', 'wb') as f:
        pickle.dump(all_preds, f)
    print('test completed in: {:2.0f}m{:2.0f}s test loss: {:4.4f} test accu: {:.4f}'
          .format(test_elapsed_time // 60, test_elapsed_time % 60, test_average_loss, test_accuracy))

print()

def main():
    _, _, _, _, test_dataset, test_num_each = get_data('train_val_test_paths_labels.pkl')

    test_model(test_dataset, test_num_each)


if __name__ == "__main__":
    main()

print('Done')
print()
