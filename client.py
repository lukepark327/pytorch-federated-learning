import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import os.path
import os

from utils import recursive_mkdir
from net import Net


class Client:
    def __init__(self,
                 trainset,
                 testset,
                 net,
                 _id=None,
                 lr: float = 1e-4,
                 betas=(0.9, 0.999),
                 weight_decay: float = 0.01,
                 with_cuda: bool = True,
                 cuda_devices=None
                 # TODO: multiple GPU
                 ):

        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        self.trainset = trainset
        if self.trainset is not None:
            self.trainloader = torch.utils.data.DataLoader(dataset=self.trainset, batch_size=4,
                                                           shuffle=True, num_workers=2)
        self.testset = testset
        if self.testset is not None:
            self.testloader = torch.utils.data.DataLoader(dataset=self.testset, batch_size=4,
                                                          shuffle=False, num_workers=2)

        # self.model = net
        self.model = net.to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for Model" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.NLLLoss(ignore_index=0)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

        assert(_id != None)
        self._id = _id  # TODO: assert error, global_id

        self.PATH = "clients/" + str(self._id) + "/"  # TODO: the other PATH for log
        recursive_mkdir(self.PATH)

    """ML"""

    def set_dataset(self, trainset, testset):  # TODO: not None
        self.trainset = trainset
        self.trainloader = torch.utils.data.DataLoader(dataset=self.trainset, batch_size=4,
                                                       shuffle=True, num_workers=2)
        self.testset = testset
        self.testloader = torch.utils.data.DataLoader(dataset=self.testset, batch_size=4,
                                                      shuffle=False, num_workers=2)

    def train(self,
              r: int,
              epochs: int = 1,
              logs: int = 2000,
              log_flag=False):

        for epoch in range(epochs):   # 데이터셋을 수차례 반복합니다.

            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
                # inputs, labels = data
                inputs, labels = data[0].to(self.device), data[1].to(self.device)  # TODO: GPU

                # 변화도(Gradient) 매개변수를 0으로 만들고
                self.optimizer.zero_grad()

                # 순전파 + 역전파 + 최적화를 한 후
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # 통계를 출력합니다.
                if log_flag:
                    running_loss += loss.item()
                    if i % logs == logs - 1:    # print every 2000 mini-batches
                        # print('[%d, %5d] loss: %.3f' %
                        #       (epoch + 1, i + 1, running_loss / logs))

                        # name = "train_loss_" + str(r) + "_" + str(epoch) + "_" + str(i) + ".log"
                        name = "train_loss.log"
                        self.log(name, r, running_loss / logs)

                        running_loss = 0.0

    def save_net(self):
        name = self.PATH + "cifar_net_" + str(self._id) + ".pth"
        torch.save(self.model.state_dict(), name)

    def load_net(self):
        name = self.PATH + "cifar_net_" + str(self._id) + ".pth"
        if os.path.isfile(name):
            print("Load net:", name)
            self.model.load_state_dict(torch.load(name))
        else:
            print("Pre-trained net does not exist")

    def eval(self,
             r: int,
             log_flag=False):
        correct = 0
        total = 0
        loss = 0
        with torch.no_grad():
            for data in self.testloader:
                # images, labels = data
                images, labels = data[0].to(self.device), data[1].to(self.device)  # TODO: GPU

                outputs = self.model(images)

                # loss += self.criterion(outputs, labels).item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # print('Accuracy of the network on the 10000 test images: %d %%' % (
        #     100 * correct / total))

        # name = "test_loss_" + str(r) + ".log"
        # self.log(name, loss / total)  # TODO: check

        acc = 100 * correct / total

        if log_flag:
            # name = "test_acc_" + str(r) + ".log"
            name = "test_acc.log"
            self.log(name, r, acc)

        return acc

    def predict(self):
        pass

    def get_weights(self):
        # return self.model.state_dict()
        params = self.model.named_parameters()
        dict_params = dict(params)

        return dict_params

    def set_weights(self, params):
        # self.model.load_state_dict(state_dict)

        my_params = self.model.named_parameters()
        dict_my_params = dict(my_params)

        for name, param in params.items():
            if name in dict_my_params:
                dict_my_params[name].data.copy_(param.data)

        self.model.load_state_dict(dict_my_params)

    def average_weights(self):
        pass

    def set_average_weights(self, paramses: list, repus: list):  # TODO: norm.
        my_params = self.model.named_parameters()
        dict_my_params = dict(my_params)

        # set zeros
        # for name in paramses[0].keys():
        #     if name in dict_my_params:
        #         dict_my_params[name].data.zero_()
        for name in paramses[0].keys():
            if name in dict_my_params:
                dict_my_params[name].data = dict_my_params[name].data * 4 / 5

        for i, repu in enumerate(repus):
            params = paramses[i]

            for name, param in params.items():
                if name in dict_my_params:
                    dict_my_params[name].data.add_(repu * param.data / 5)

        self.model.load_state_dict(dict_my_params)

    """DAG"""

    def select_node(self):
        pass

    def test_node(self):
        pass

    def create_node(self):
        pass

    """Vis"""

    def log(self, name: str, r: int, data):
        with open(self.PATH + name, "a") as f:
            f.write("%d, %f\n" % (r, data))


if __name__ == "__main__":
    """Preprocess"""
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    # print(len(trainset))  # 50000
    # print(len(testset))  # 10000

    """random split"""
    splited_trainset = torch.utils.data.random_split(trainset, [15000, 25000, 10000])
    splited_testset = torch.utils.data.random_split(testset, [2000, 6000, 2000])

    # print(len(splited_trainset[0]), len(splited_trainset[1]), len(splited_trainset[2]))

    clients = []
    for i in range(3):
        clients.append(Client(
            trainset=splited_trainset[i],
            testset=splited_testset[i],
            net=Net(),
            _id=i
        ))

    clients[0].train(r=0, epochs=1)
    clients[1].set_weights(clients[0].get_weights())
    clients[2].set_weights(clients[0].get_weights())

    clients[0].eval(r=0)
    clients[1].eval(r=0)
    clients[2].eval(r=0)

    clients[0].set_average_weights(
        [clients[1].get_weights(), clients[2].get_weights()],
        [0.9, 0.1])

    clients[0].eval(r=1)
    clients[1].eval(r=1)
    clients[2].eval(r=1)
