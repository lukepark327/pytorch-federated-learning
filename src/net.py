"""Ref.
https://tutorials.pytorch.kr/beginner/blitz/cifar10_tutorial.html
https://tutorials.pytorch.kr/beginner/blitz/data_parallel_tutorial.html
"""
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import os.path


# """show images"""
# # 이미지를 보여주기 위한 함수
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # """GPU"""
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # # CUDA 기기가 존재한다면, 아래 코드가 CUDA 장치를 출력합니다:
    # print(device)

    """Preprocess"""
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # # 학습용 이미지를 무작위로 가져오기
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()

    # # 이미지 보여주기
    # imshow(torchvision.utils.make_grid(images))
    # # 정답(label) 출력
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    """Net"""
    # net = Net()
    net = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
    net.classifier = nn.Linear(in_features=1024, out_features=10, bias=True)  # diff. output features


    # net.to(device)  # TODO: GPU

    """load"""
    PATH = './cifar_net.pth'
    if os.path.isfile(PATH):
        print("Load net:", PATH)
        net.load_state_dict(torch.load(PATH))
    else:
        print("Pre-trained net does not exist")

    """loss and optimizer"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    """training"""
    for epoch in range(1):   # 데이터셋을 수차례 반복합니다.

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
            inputs, labels = data
            # inputs, labels = data[0].to(device), data[1].to(device)  # TODO: GPU

            # 변화도(Gradient) 매개변수를 0으로 만들고
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화를 한 후
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 통계를 출력합니다.
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    """model save"""
    torch.save(net.state_dict(), PATH)

    """eval"""
    # dataiter = iter(testloader)
    # images, labels = dataiter.next()

    # # 이미지를 출력합니다.
    # imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # outputs = net(images)
    # _, predicted = torch.max(outputs, 1)
    # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
    #                             for j in range(4)))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

    # # per class
    # class_correct = list(0. for i in range(10))
    # class_total = list(0. for i in range(10))
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs, 1)
    #         c = (predicted == labels).squeeze()
    #         for i in range(4):
    #             label = labels[i]
    #             class_correct[label] += c[i].item()
    #             class_total[label] += 1

    # for i in range(10):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[i], 100 * class_correct[i] / class_total[i]))
