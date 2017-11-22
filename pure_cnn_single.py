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
optimizer_choice = 1  # 0 for SGD, 1 for Adam89

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
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 7)

    def forward(self, x):
        # x = self.share.forward(x)
        x = x.view(-1, 2048)
        y = self.fc1(x)
        y = self.fc2(y)
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

    train_idx = [i for i in range(num_train)]
    np.random.seed(0)
    np.random.shuffle(train_idx)
    val_idx = [i for i in range(num_val)]

    num_train_all = len(train_idx)
    num_val_all = len(val_idx)

    print('num of trainset:', num_train)
    print('num of all train samples:', num_train_all)
    print('train batch size:', train_batch_size)
    print('sequence length:', sequence_length)
    print('num of gpu:', num_gpu)

    print('num of valset:', num_val)
    print('num of all val samples:', num_val_all)

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=train_sampler,
        # shuffle=True,
        num_workers=8,
        pin_memory=False
    )
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_idx)
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        sampler=val_sampler,
        # shuffle=True,
        num_workers=8,
        pin_memory=False
    )

    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7)
    # model = my_resnet()
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

            outputs = model.forward(inputs)

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
    torch.save(model, '20171121_epoch_25_pure_cnn_single_adam.pth')
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

    # _, _, _, _, test_dataset, test_num_each = get_data('train_val_test_paths_labels.pkl')
    # test_model(test_dataset)


if __name__ == "__main__":
    main()

print('Done')
print()
