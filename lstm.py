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


# batch_size 要整除gpu个数 以及sequence长度
sequence_length = 4
train_batch_size = 100
val_batch_size = 8
lstm_in_dim = 2048
lstm_out_dim = 512
optimizer_choice = 0  # 0 for SGD, 1 for Adam89


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



def train_model(train_dataset, train_num_each, val_dataset, val_num_each):
    num_train = len(train_dataset)
    num_val = len(val_dataset)
    print('num of train:', num_train)
    train_count = 0
    for i in range(len(train_num_each)):
        train_count += train_num_each[i]
    print('vertify num of train:', train_count)
    print('num of valid:', num_val)
    val_count = 0
    for i in range(len(val_num_each)):
        val_count += val_num_each[i]
    print('vertify num of valid:', val_count)

    train_useful_start_idx = get_useful_start_idx(sequence_length, train_num_each)

    val_useful_start_idx = get_useful_start_idx(sequence_length, val_num_each)
    print('num of useful train start idx:', len(train_useful_start_idx))
    print('the last idx of train start idx:', train_useful_start_idx[-1])

    print('num of useful valid start idx:', len(val_useful_start_idx))
    print('the last idx of train start idx:', val_useful_start_idx[-1])

    num_train_we_use = len(train_useful_start_idx) // (train_batch_size // sequence_length) * (
        train_batch_size // sequence_length)
    num_val_we_use = len(val_useful_start_idx) // (train_batch_size // sequence_length) * (
        train_batch_size // sequence_length)
    # num_train_we_use = 800
    # num_val_we_use = 80

    train_we_use_start_idx = train_useful_start_idx[0:num_train_we_use]
    val_we_use_start_idx = val_useful_start_idx[0:num_val_we_use]

    # 此处可以shuffle train 的idx


    # num_train_truth = 12000
    # num_train_truth = (num_train + 1 - sequence_length) // (train_batch_size // sequence_length) * (train_batch_size // sequence_length)
    # train_true_start_idx = list(range((num_train_truth)))
    # train_true_start_idx = list(range(100))
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
    print('num of trainset:', num_train)
    print('num of train samples we use:', num_train_we_use)
    print('num of all train samples:', num_train_all)
    print('train batch size:', train_batch_size)
    print('sequence length:', sequence_length)
    print('num of gpu:', num_gpu)

    print('num of valset:', num_val)
    print('num of val samples we use:', num_val_we_use)
    print('num of all val samples:', num_val_all)

    print(train_idx[0:20])
    print(val_idx[0:210])

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
    # model = models.resnet50(pretrained=True)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 7)
    model = my_resnet()
    if use_gpu:
        model = model.cuda()
    model = DataParallel(model)
    criterion = nn.CrossEntropyLoss(size_average=False)
    # 要先将model转换到cuda, 再提供optimizer
    # for parameter in model.parameters():
    #     print(parameter)
    if optimizer_choice == 0:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        # exp_scheduler = ReduceLROnPlateau(optimizer, 'min') val loss 调用 expscheduler.step(loss)
    elif optimizer_choice == 1:
        optimizer = optim.Adam(model.parameters())

    best_model_wts = model.state_dict()
    best_val_accuracy = 0.0
    correspond_train_acc = 0.0

    epochs = 25
    all_info = []
    all_train_accuracy = []
    all_train_loss = []
    all_val_accuracy = []
    all_val_loss = []
    for epoch in range(epochs):

        model.train()
        # if optimizer_choice == 0:
        # exp_lr_scheduler.step()
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
            # 如果不在内部调用, 会出现显存持续增长的问题, 还不知道为什么

            outputs = model.forward(inputs)
            # print(outputs.size())

            # print(outputs.size())

            # output of lstm 非 congiguous
            # print(model.module.forward_batch_size)
            # print(outputs.size())
            # print(outputs.requires_grad)
            # print(outputs.size())
            # print(outputs.data[0])
            # print(labels.data[0])
            # print(labels.size())
            # print(outputs.size())
            _, preds = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            # print(loss)
            loss.backward()
            # count +=1
            optimizer.step()
            train_loss += loss.data[0]
            train_corrects += torch.sum(preds == labels.data)
            # print(train_corrects)
        train_elapsed_time = time.time() - train_start_time
        train_accuracy = train_corrects / num_train_all
        train_average_loss = train_loss / num_train_all

        # train_average_loss = train_loss / num_train
        # print('accuracy', train_accuracy)
        # print('train loss: {:.4f} accuracy: {:.4f}'.format(train_average_loss, train_accuracy))

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

            model.module.hidden = model.module.init_hidden(val_batch_size // sequence_length // num_gpu)
            # 如果不在内部调用, 会出现显存持续增长的问题, 还不知道为什么

            outputs = model.forward(inputs)

            # outputs = outputs.contiguous().view(num_gpu, sequence_length, -1, 7)
            # outputs = outputs.permute(0, 2, 1, 3)
            # outputs = outputs.contiguous().view((val_batch_size, 7))

            _, preds = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            val_loss += loss.data[0]
            val_corrects += torch.sum(preds == labels.data)
            # print(val_corrects)
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
    torch.save(model, '20171118_epoch_25_cnn_lstm_fc_length_4_sgd_on_loss.pth')
    all_info.append(all_train_accuracy)
    all_info.append(all_train_loss)
    all_info.append(all_val_accuracy)
    all_info.append(all_val_loss)
    with open('20171121_epoch_25_pure_cnn_single_adam.pkl', 'wb') as f:
        pickle.dump(all_info, f)
    print()

def main():
    train_dataset, train_num_each, val_dataset, val_num_each, _, _ = get_data('train_val_test_paths_labels.pkl')
    train_model(train_dataset, train_num_each, val_dataset, val_num_each)

if __name__ == "__main__":
    main()

print('Done')
print()
