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
                 loader = pil_loader):
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

sequence_length = 3
train_batch_size = 90
lstm_in_dim = 2048

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
        self.lstm = nn.LSTM(lstm_in_dim, 7)
        self.hidden = self.init_hidden()
        # print(len(self.lstm.all_weights))
        # print(len(self.lstm.all_weights[0]))

        init.xavier_normal(self.lstm.all_weights[0][0])
        init.xavier_normal(self.lstm.all_weights[0][1])
        # self.count = 0
        # 多GPU时候.这种赋值方式不成功, 所以尽量取能整除的batch
        # self.forward_batch_size = 0
    def init_hidden(self, hidden_batch_size = 1):
        if use_gpu:
            return (Variable(torch.zeros(1, hidden_batch_size, 7).cuda()),
                    Variable(torch.zeros(1, hidden_batch_size, 7).cuda()))
        else:
            return (Variable(torch.zeros(1, hidden_batch_size, 7)),
               Variable(torch.zeros(1, hidden_batch_size, 7)))

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
        y, self.hidden = self.lstm(x, self.hidden)
        return y

def get_data(data_path):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)
    train_paths = train_test_paths_labels[0]
    test_paths = train_test_paths_labels[1]
    train_labels = train_test_paths_labels[2]
    test_labels = train_test_paths_labels[3]
    # print(len(train_paths))
    # print(len(test_paths))
    # print(train_labels.shape)
    # print(test_labels.shape)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    train_transforms = transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    test_transforms = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    train_dataset = CholecDataset(train_paths, train_labels, train_transforms)
    test_dataset = CholecDataset(test_paths, test_labels, test_transforms)
    return train_dataset, test_dataset

def train_model(train_dataset):
    num_train = len(train_dataset)
    num_train_valid = 1200
    # num_train_valid = (num_train + 1 - sequence_length) // (train_batch_size // sequence_length) * (train_batch_size // sequence_length)
    train_valid_start_idx = list(range((num_train_valid)))
    # train_valid_start_idx = list(range(100))
    np.random.seed(0)
    np.random.shuffle(train_valid_start_idx)
    train_idx = []

    for i in range(num_train_valid):
    # for i in range(100):
        train_idx.append(train_valid_start_idx[i])
        train_idx.append(train_valid_start_idx[i] + 1)
        train_idx.append(train_valid_start_idx[i] + 2)
    num_train_all = len(train_idx)
    print('num of trainset:', num_train)
    print('num of valid samples:', num_train_valid)
    print('num of all samples:', num_train_all)
    print('train batch size:', train_batch_size)
    print('sequence length:', sequence_length)
    print('num of gpu:', num_gpu)

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=train_sampler,
        # shuffle=True,
        num_workers=9,
        pin_memory=False
    )
    batch_size = train_loader.batch_size

    # model = models.resnet50(pretrained=True)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 7)
    model = my_resnet()
    if use_gpu:
        model = model.cuda()
    model = DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    epoches = 25
    for epoch in range(epoches):
        model.train()
        exp_lr_scheduler.step()
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
            model.module.hidden = model.module.init_hidden(train_batch_size // sequence_length // num_gpu)

            # print(outputs.size())

            # print(outputs.size())

            # output of lstm 非 congiguous
            # print(model.module.forward_batch_size)
            # print(outputs.size())
            outputs = model.forward(inputs)
            # print(outputs.size())
            # print(labels.size())
            # labels = labels.contiguous().view(train_batch_size//sequence_length , sequence_length, 7)
            # labesl = labels.permute()
            outputs = outputs.contiguous().view(num_gpu, sequence_length, -1, 7)
            outputs = outputs.permute(0, 2, 1, 3)
            outputs = outputs.contiguous().view((train_batch_size, 7))
            # print(outputs.requires_grad)
            # print(outputs.size())
            _,preds = torch.max(outputs.data, 1)
            # print(outputs.size())
            # print(labels.size())
            loss = criterion(outputs, labels)
            # print(loss)
            loss.backward()
            optimizer.step()
            train_loss += loss.data[0]
            train_corrects += torch.sum(preds == labels.data)
            # print(train_corrects)
        train_elapsed_time = time.time()- train_start_time

        print('epoch: {:4d} train completed in: {:.0f}m{:.0f}s'.format(epoch, train_elapsed_time // 60, train_elapsed_time % 60))

        # train_average_loss = train_loss / num_train
        train_accuracy = train_corrects / num_train_all
        print('accuracy', train_accuracy)
        # print('train loss: {:.4f} accuracy: {:.4f}'.format(train_average_loss, train_accuracy))
    torch.save(model, '20171114_epoch_25_lstm.pth')

def test_model(test_dataset):

    test_start_time = time.time()
    test_loader = DataLoader(
        test_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )
    num_test = len(test_dataset)
    batch_size = test_loader.batch_size
    model = torch.load('20171113_epoch_25.pkl')
    model.eval()
    test_loss = 0.0
    test_corrects = 0
    if use_gpu:
        model = model.cuda()
    for data in test_loader:
        inputs, labels_1, labels_2 = data
        if use_gpu:
            inputs = Variable(inputs.cuda(), volatile=True)
            labels = Variable(labels_2.cuda(), volatile=True)
        else:
            inputs = Variable(inputs, volatile=True)
            labels = Variable(labels_2, volatile=True)
        outputs = model.forward(inputs)
        test_corrects += torch.sum(preds == labels.data)
    test_elapsed_time = time.time() - test_start_time
    print('test completed in: {:.0f}m{:.0f}s'.format(
            test_elapsed_time // 60, test_elapsed_time % 60))

    test_accuracy = test_corrects / num_test
    print('accuracy', test_accuracy)

def main():

    train_dataset, test_dataset = get_data('train_test_paths_labels.pkl')

    train_model(train_dataset)
    # test_model(test_dataset)
if __name__ == "__main__":
    main()

print('Done')
print()
