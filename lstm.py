import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import time
from torch.nn import DataParallel
import os
import time
import pickle
import argparse
from PIL import Image

parser = argparse.ArgumentParser(description='lstm Training')
parser.add_argument('-g', '--gpu', default=1, nargs='+', type=int,help='index of gpu to use')
parser.add_argument('-s', '--seq', default=4, type=int, help='sequence length')
parser.add_argument('-t', '--train', default=100, type=int, help='train batch size')
parser.add_argument('-v', '--val', default=8, type=int, help='valid batch size')
parser.add_argument('-o', '--opt', default=0, type=int, help='0 for sgd 1 for adam')
parser.add_argument('-e', '--epo', default=25, type=int, help='epochs to train and val')

args = parser.parse_args()
gpu_usg = ",".join(list(map(str, args.gpu)))
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg
sequence_length = args.seq
train_batch_size = args.train
val_batch_size = args.val
optimizer_choice = args.opt
epochs = args.epo

num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()

lstm_in_dim = 2048
lstm_out_dim = 512


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

        init.xavier_normal(self.lstm.all_weights[0][0])
        init.xavier_normal(self.lstm.all_weights[0][1])

    def forward(self, x):
        x = self.share.forward(x)
        x = x.view(-1, 2048)
        x = x.view(-1, sequence_length, lstm_in_dim)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, lstm_out_dim)
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


def train_model(train_dataset, train_num_each, val_dataset, val_num_each):
    num_train = len(train_dataset)
    num_val = len(val_dataset)

    train_useful_start_idx = get_useful_start_idx(sequence_length, train_num_each)

    val_useful_start_idx = get_useful_start_idx(sequence_length, val_num_each)
    print('num of useful train start idx:', len(train_useful_start_idx))
    print('the last idx of train start idx:', train_useful_start_idx[-1])
    print('num of useful valid start idx:', len(val_useful_start_idx))
    print('the last idx of train start idx:', val_useful_start_idx[-1])

    # num_train_we_use = len(train_useful_start_idx) // num_gpu * num_gpu
    # num_val_we_use = len(val_useful_start_idx) // num_gpu * num_gpu
    num_train_we_use = 8000
    num_val_we_use = 800

    train_we_use_start_idx = train_useful_start_idx[0:num_train_we_use]
    val_we_use_start_idx = val_useful_start_idx[0:num_val_we_use]

    # 此处可以shuffle train 的idx
    np.random.seed(0)
    np.random.shuffle(train_we_use_start_idx)
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
    print('num of gpu:', num_gpu)
    print('sequence length:', sequence_length)
    print('num of trainset:', num_train)
    print('num train we use:', num_train_we_use)
    print('num train all:', num_train_all)
    print('train batch size:', train_batch_size)

    print('num of valset:', num_val)
    print('num val we use:', num_val_we_use)
    print('num val we use:', num_val_all)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=train_idx,
        # shuffle=True,
        num_workers=8,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        sampler=val_idx,
        # shuffle=True,
        num_workers=8,
        pin_memory=False
    )
    model = resnet_lstm()
    if use_gpu:
        model = model.cuda()

    model = DataParallel(model)
    criterion = nn.CrossEntropyLoss(size_average=False)

    if optimizer_choice == 0:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    elif optimizer_choice == 1:
        optimizer = optim.Adam(model.parameters())

    best_model_wts = model.state_dict()
    best_val_accuracy = 0.0
    correspond_train_acc = 0.0

    all_info = []
    all_train_accuracy = []
    all_train_loss = []
    all_val_accuracy = []
    all_val_loss = []

    for epoch in range(epochs):

        model.train()

        train_loss = 0.0
        train_corrects = 0
        train_start_time = time.time()
        for data in train_loader:
            inputs, labels_1, labels_2 = data
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels_2.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels_2)
            optimizer.zero_grad()  # 如果optimizer(net.parameters()), 那么效果和net.zero_grad()一样

            outputs = model.forward(inputs)
            _, preds = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
            train_corrects += torch.sum(preds == labels.data)
        train_elapsed_time = time.time() - train_start_time
        train_accuracy = train_corrects / num_train_all
        train_average_loss = train_loss / num_train_all

        # begin eval

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_start_time = time.time()
        for data in val_loader:
            inputs, labels_1, labels_2 = data
            if use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels_2.cuda())
            else:
                inputs = Variable(inputs)
                labels = Variable(labels_2)

            outputs = model.forward(inputs)

            _, preds = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            val_loss += loss.data[0]
            val_corrects += torch.sum(preds == labels.data)
        val_elapsed_time = time.time() - val_start_time
        val_accuracy = val_corrects / num_val_all
        val_average_loss = val_loss / num_val_all
        print('epoch: {:4d} train completed in: {:2.0f}m{:2.0f}s  train loss: {:4.4f} train accu: {:.4f}'
              'valid completed in: {:.0f}m{:.0f}s '
              'valid loss: {:4.4f} valid accu: {:.4f}'.format(epoch, train_elapsed_time // 60, train_elapsed_time % 60,
                                                              train_average_loss, train_accuracy,
                                                              val_elapsed_time // 60, val_elapsed_time % 60,
                                                              val_average_loss, val_accuracy))

        if optimizer_choice == 0:
            exp_lr_scheduler.step(val_average_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            correspond_train_acc = train_accuracy
            best_model_wts = model.state_dict()
        if val_accuracy == best_val_accuracy:
            if train_accuracy > correspond_train_acc:
                correspond_train_acc = train_accuracy
                best_model_wts = model.state_dict()

        all_train_loss.append(train_average_loss)
        all_train_accuracy.append(train_accuracy)
        all_val_loss.append(val_average_loss)
        all_val_accuracy.append(val_accuracy)

    print('best accuracy: {:.4f} cor train accu: {:.4f}'.format(best_val_accuracy, correspond_train_acc))
    model.load_state_dict(best_model_wts)
    save_val = int("{:4d}".format(best_val_accuracy * 10000))
    save_train= int("{:4d}".format(correspond_train_acc * 10000))
    model_name = "lstm_epoch" + str(epochs) + "_length" + str(
        sequence_length) + "_opt_" + str(optimizer_choice) + "_batch_" + str(train_batch_size) + "_train_" + str(
        save_train) + "_val_" + str(save_val) + ".pth"

    torch.save(model, model_name)
    all_info.append(all_train_accuracy)
    all_info.append(all_train_loss)
    all_info.append(all_val_accuracy)
    all_info.append(all_val_loss)
    record_name = "lstm_epoch" + str(epochs) + "_length" + str(
        sequence_length) + "_opt_" + str(optimizer_choice) + "_batch_" + str(train_batch_size) + "_train_" + str(
        save_train) + "_val_" + str(save_val) + ".pkl"
    with open(record_name, 'wb') as f:
        pickle.dump(all_info, f)
    print()

def main():
    train_dataset, train_num_each, val_dataset, val_num_each, _, _ = get_data('train_val_test_paths_labels.pkl')
    train_model(train_dataset, train_num_each, val_dataset, val_num_each)

if __name__ == "__main__":
    main()

print('Done')
print()
