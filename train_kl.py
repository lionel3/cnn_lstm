import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
import os
from PIL import Image
import time
import pickle
import numpy as np
import argparse
import copy
from torchvision.transforms import Lambda
import random

parser = argparse.ArgumentParser(description='kl training')
parser.add_argument('-g', '--gpu', default=[1], nargs='+', type=int, help='index of gpu to use, default 1')
parser.add_argument('-s', '--seq', default=4, type=int, help='sequence length, default 4')
parser.add_argument('-t', '--train', default=100, type=int, help='train batch size, default 100')
parser.add_argument('-v', '--val', default=8, type=int, help='valid batch size, default 8')
parser.add_argument('-o', '--opt', default=1, type=int, help='0 for sgd 1 for adam, default 1')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt, default 1')
parser.add_argument('-e', '--epo', default=25, type=int, help='epochs to train and val, default 25')
parser.add_argument('-w', '--work', default=2, type=int, help='num of workers to use, default 2')
parser.add_argument('-f', '--flip', default=0, type=int, help='0 for not flip, 1 for flip, default 0')
parser.add_argument('-c', '--crop', default=1, type=int, help='0 rand, 1 cent, 5 five_crop, 10 ten_crop, default 1')
parser.add_argument('-l', '--lr', default=1e-3, type=float, help='learning rate for optimizer, default 1e-3')

args = parser.parse_args()

gpu_usg = ",".join(list(map(str, args.gpu)))
sequence_length = args.seq
train_batch_size = args.train
val_batch_size = args.val
optimizer_choice = args.opt
multi_optim = args.multi
epochs = args.epo
workers = args.work
use_flip = args.flip
crop_type = args.crop
learing_rate = args.lr

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg
num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()

print('number of gpu   : {:6d}'.format(num_gpu))
print('sequence length : {:6d}'.format(sequence_length))
print('train batch size: {:6d}'.format(train_batch_size))
print('valid batch size: {:6d}'.format(val_batch_size))
print('optimizer choice: {:6d}'.format(optimizer_choice))
print('multiple optim  : {:6d}'.format(multi_optim))
print('num of epochs   : {:6d}'.format(epochs))
print('num of workers  : {:6d}'.format(workers))
print('test crop type  : {:6d}'.format(crop_type))
print('whether to flip : {:6d}'.format(use_flip))
print('learning rate   : {:.4f}'.format(learing_rate))


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class RandomHorizontalFlip(object):
    def __init__(self):
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        self.count += 1
        random.seed(seed)
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


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


class multi_lstm(torch.nn.Module):
    def __init__(self):
        super(multi_lstm, self).__init__()
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
        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.fc = nn.Linear(512, 7)
        self.fc2 = nn.Linear(2048, 7)
        init.xavier_normal(self.lstm.all_weights[0][0])
        init.xavier_normal(self.lstm.all_weights[0][1])
        init.xavier_uniform(self.fc.weight)
        init.xavier_uniform(self.fc2.weight)

    def forward(self, x):
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        z = self.fc2(x)
        x = x.view(-1, sequence_length, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
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

    print('train_paths  : {:6d}'.format(len(train_paths)))
    print('train_labels : {:6d}'.format(len(train_labels)))
    print('valid_paths  : {:6d}'.format(len(val_paths)))
    print('valid_labels : {:6d}'.format(len(val_labels)))
    print('test_paths   : {:6d}'.format(len(test_paths)))
    print('test_labels  : {:6d}'.format(len(test_labels)))

    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    if use_flip == 0:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
        ])
    elif use_flip == 1:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(224),
            RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
        ])

    if crop_type == 0:
        test_transforms = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
        ])
    elif crop_type == 1:
        test_transforms = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
        ])
    elif crop_type == 5:
        test_transforms = transforms.Compose([
            transforms.FiveCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])(crop) for crop in crops]))
        ])
    elif crop_type == 10:
        test_transforms = transforms.Compose([
            transforms.TenCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(
                lambda crops: torch.stack(
                    [transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])(crop) for crop in crops]))
        ])

    train_dataset = CholecDataset(train_paths, train_labels, train_transforms)
    val_dataset = CholecDataset(val_paths, val_labels, test_transforms)
    test_dataset = CholecDataset(test_paths, test_labels, test_transforms)

    return train_dataset, train_num_each, val_dataset, val_num_each, test_dataset, test_num_each


def train_model(train_dataset, train_num_each, val_dataset, val_num_each):

    num_train = len(train_dataset)
    num_val = len(val_dataset)

    train_useful_start_idx = get_useful_start_idx(sequence_length, train_num_each)
    val_useful_start_idx = get_useful_start_idx(sequence_length, val_num_each)

    num_train_we_use = len(train_useful_start_idx) // num_gpu * num_gpu
    num_val_we_use = len(val_useful_start_idx) // num_gpu * num_gpu
    # num_train_we_use = 800
    # num_val_we_use = 800

    train_we_use_start_idx = train_useful_start_idx[0:num_train_we_use]
    val_we_use_start_idx = val_useful_start_idx[0:num_val_we_use]

    train_idx = []
    for i in range(num_train_we_use):
        for j in range(sequence_length):
            train_idx.append(train_we_use_start_idx[i] + j)

    val_idx = []
    for i in range(num_val_we_use):
        for j in range(sequence_length):
            val_idx.append(val_we_use_start_idx[i] + j)

    num_train_all = len(train_idx)
    num_val_all = len(val_idx)

    print('num train start idx : {:6d}'.format(len(train_useful_start_idx)))
    print('last idx train start: {:6d}'.format(train_useful_start_idx[-1]))
    print('num of train dataset: {:6d}'.format(num_train))
    print('num of train we use : {:6d}'.format(num_train_we_use))
    print('num of all train use: {:6d}'.format(num_train_all))
    print('num valid start idx : {:6d}'.format(len(val_useful_start_idx)))
    print('last idx valid start: {:6d}'.format(val_useful_start_idx[-1]))
    print('num of valid dataset: {:6d}'.format(num_val))
    print('num of valid we use : {:6d}'.format(num_val_we_use))
    print('num of all valid use: {:6d}'.format(num_val_all))

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=train_idx,
        num_workers=workers,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        sampler=val_idx,
        num_workers=workers,
        pin_memory=False
    )

    model = multi_lstm()
    model = DataParallel(model)
    model.load_state_dict(torch.load('cnn_lstm_epoch_25_length_4_opt_1_mulopt_1_flip_0_crop_1_batch_200_train1_9998_train2_9987_val1_9731_val2_8752.pth'))

    kl_fc_p2t = nn.Linear(7, 7)
    # kl_fc_t2p = nn.Linear(7, 7)

    for param in model.module.parameters():
        param.requires_grad = False
    for param in kl_fc_p2t.parameters():
        param.requires_grad = True

    if use_gpu:
        kl_fc_p2t = kl_fc_p2t.cuda()

    criterion_1 = nn.BCEWithLogitsLoss(size_average=False)
    criterion_2 = nn.CrossEntropyLoss(size_average=False)
    criterion_3 = nn.KLDivLoss(size_average=False)
    sig_f = nn.Sigmoid()

    if multi_optim == 0:
        if optimizer_choice == 0:
            optimizer = optim.SGD(kl_fc_p2t.parameters(), lr=learing_rate, momentum=0.9)
            exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.Adam(kl_fc_p2t.parameters(), lr=learing_rate)
    elif multi_optim == 1:
        if optimizer_choice == 0:
            optimizer = optim.SGD(kl_fc_p2t.parameters(), lr=learing_rate, momentum=0.9)
            exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        elif optimizer_choice == 1:
            optimizer = optim.Adam(kl_fc_p2t.parameters(), lr=learing_rate)

    for epoch in range(epochs):
        np.random.shuffle(train_we_use_start_idx)
        train_idx = []
        for i in range(num_train_we_use):
            for j in range(sequence_length):
                train_idx.append(train_we_use_start_idx[i] + j)

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            sampler=train_idx,
            num_workers=workers,
            pin_memory=False
        )

        kl_fc_p2t.train()

        train_loss_1 = 0.0
        train_loss_2 = 0.0
        train_loss_3 = 0.0
        train_corrects_1 = 0
        train_corrects_2 = 0

        train_start_time = time.time()
        for data in train_loader:
            inputs, labels_1, labels_2 = data
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels_1 = Variable(labels_1.cuda())
                labels_2 = Variable(labels_2.cuda())
            else:
                inputs = Variable(inputs)
                labels_1 = Variable(labels_1)
                labels_2 = Variable(labels_2)

            optimizer.zero_grad()

            outputs_1, outputs_2 = model.forward(inputs)

            _, preds_2 = torch.max(outputs_2.data, 1)

            sig_out = outputs_1.data.cpu()
            sig_out = sig_f(sig_out)
            preds_1 = torch.ByteTensor(sig_out > 0.5)
            preds_1 = preds_1.long()
            train_corrects_1 += torch.sum(preds_1 == labels_1.data.cpu())
            labels_1 = Variable(labels_1.data.float())
            loss_1 = criterion_1(outputs_1, labels_1)
            loss_2 = criterion_2(outputs_2, labels_2)


            kl_softmax = nn.Softmax().cuda()
            kl_output_1 = kl_softmax(outputs_1)
            kl_output_2 = kl_softmax(kl_fc_p2t(outputs_2))
            kl_output_1 = Variable(kl_output_1.data, requires_grad=False)
            loss_3 = torch.abs(criterion_3(kl_output_2, kl_output_1))
            loss = loss_3
            loss.backward()
            optimizer.step()
            train_loss_1 += loss_1.data[0]
            train_loss_2 += loss_2.data[0]
            train_loss_3 += loss_3.data[0]
            train_corrects_2 += torch.sum(preds_2 == labels_2.data)

        train_elapsed_time = time.time() - train_start_time
        train_accuracy_1 = train_corrects_1 / num_train_all / 7
        train_accuracy_2 = train_corrects_2 / num_train_all
        train_average_loss_1 = train_loss_1 / num_train_all / 7
        train_average_loss_2 = train_loss_2 / num_train_all
        train_average_loss_3 = train_loss_3 / num_train_all

        # begin eval

        kl_fc_p2t.eval()

        val_loss_1 = 0.0
        val_loss_2 = 0.0
        val_loss_3 = 0.0
        val_corrects_1 = 0
        val_corrects_2 = 0

        val_start_time = time.time()
        for data in val_loader:
            inputs, labels_1, labels_2 = data
            labels_2 = labels_2[(sequence_length - 1):: sequence_length]
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels_1 = Variable(labels_1.cuda())
                labels_2 = Variable(labels_2.cuda())
            else:
                inputs = Variable(inputs)
                labels_1 = Variable(labels_1)
                labels_2 = Variable(labels_2)

            if crop_type == 0 or crop_type == 1:
                outputs_1, outputs_2 = model.forward(inputs)
            elif crop_type == 5:
                inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
                inputs = inputs.view(-1, 3, 224, 224)
                outputs_1, outputs_2 = model.forward(inputs)
                outputs_1 = outputs_1.view(5, -1, 7)
                outputs_1 = torch.mean(outputs_1, 0)
                outputs_2 = outputs_2.view(5, -1, 7)
                outputs_2 = torch.mean(outputs_2, 0)
            elif crop_type == 10:
                inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
                inputs = inputs.view(-1, 3, 224, 224)
                outputs_1, outputs_2 = model.forward(inputs)
                outputs_1 = outputs_1.view(10, -1, 7)
                outputs_1 = torch.mean(outputs_1, 0)
                outputs_2 = outputs_2.view(10, -1, 7)
                outputs_2 = torch.mean(outputs_2, 0)

            outputs_2 = outputs_2[sequence_length - 1::sequence_length]
            _, preds_2 = torch.max(outputs_2.data, 1)

            sig_out = outputs_1.data.cpu()
            sig_out = sig_f(sig_out)
            preds_1 = torch.ByteTensor(sig_out > 0.5)
            preds_1 = preds_1.long()
            val_corrects_1 += torch.sum(preds_1 == labels_1.data.cpu())
            labels_1 = Variable(labels_1.data.float())
            loss_1 = criterion_1(outputs_1, labels_1)
            val_loss_1 += loss_1.data[0]

            loss_2 = criterion_2(outputs_2, labels_2)
            val_loss_2 += loss_2.data[0]
            val_corrects_2 += torch.sum(preds_2 == labels_2.data)

            kl_softmax = nn.Softmax().cuda()
            kl_output_1 = kl_softmax(outputs_1[sequence_length - 1::sequence_length])
            kl_output_2 = kl_softmax(kl_fc_p2t(outputs_2))
            kl_output_1 = Variable(kl_output_1.data, requires_grad=False)
            loss_3 = torch.abs(criterion_3(kl_output_2, kl_output_1))
            val_loss_3 += loss_3.data[0]

        val_elapsed_time = time.time() - val_start_time
        val_accuracy_1 = val_corrects_1 / (num_val_all * 7)
        val_accuracy_2 = val_corrects_2 / num_val_we_use
        val_average_loss_1 = val_loss_1 / (num_val_all * 7)
        val_average_loss_2 = val_loss_2 / num_val_we_use
        val_average_loss_3 = val_loss_3 / num_val_we_use

        print('epoch: {:4d}'
              ' train time: {:2.0f}m{:2.0f}s'
              ' train accu_1: {:.4f}'
              ' train accu_2: {:.4f}'
              ' train loss_1: {:4.4f}'
              ' train loss_2: {:4.4f}'
              ' train loss_3: {:4.4f}'
              .format(epoch,
                      train_elapsed_time // 60,
                      train_elapsed_time % 60,
                      train_accuracy_1,
                      train_accuracy_2,
                      train_average_loss_1,
                      train_average_loss_2,
                      train_average_loss_3))
        print('epoch: {:4d}'
              ' valid time: {:2.0f}m{:2.0f}s'
              ' valid accu_1: {:.4f}'
              ' valid accu_2: {:.4f}'
              ' valid loss_1: {:4.4f}'
              ' valid loss_2: {:4.4f}'
              ' valid loss_3: {:4.4f}'
              .format(epoch,
                      val_elapsed_time // 60,
                      val_elapsed_time % 60,
                      val_accuracy_1,
                      val_accuracy_2,
                      val_average_loss_1,
                      val_average_loss_2,
                      val_average_loss_3))

        if optimizer_choice == 0:
            exp_lr_scheduler.step(val_average_loss_3)

        print(kl_fc_p2t.weight)

def main():

    train_dataset, train_num_each, val_dataset, val_num_each, _, _ = get_data('train_val_test_paths_labels.pkl')
    train_model(train_dataset, train_num_each, val_dataset, val_num_each)


if __name__ == "__main__":
    main()

print('Done')
print()
