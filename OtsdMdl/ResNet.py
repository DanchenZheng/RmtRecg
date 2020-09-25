#
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as xfrm
#
from torch.utils.data import Dataset

# device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
EPOCHS = 100
batch_size = 2048
learning_rate = 0.01
classes = 10

# Image preprocessing modules
transform = xfrm.Compose([
    # xfrm.Pad(4),
    xfrm.RandomHorizontalFlip(),
    # xfrm.RandomCrop(32),
    xfrm.ToTensor(),
])

# cifar10 32*32*3
train_dataset = torchvision.datasets.CIFAR10(root='../../cifar10_data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='../../cifar10_data', train=False, transform=xfrm.ToTensor())

trainloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

save_path = "F:\\Remote\\TesRecg\\OtsdMdl"

# # 训练集、测试集加载
# train_dataset = torchvision.datasets.MNIST(
#     root='./mnist',
#     train=True,
#     transform=transform,
#     download=False
# )
# trainloader = Dataset.DataLoader(dataset=train_dataset, BATCH_SIZE=batch_size, shuffle=True)
# test_dataset = torchvision.datasets.MNIST(root='./mnist/', train=False,  transform=transform)
# testloader = Dataset.DataLoader(dataset=test_dataset, BATCH_SIZE=batch_size, shuffle=True)


#
class ResBlk(nn.Module):
    def __init__(self, in_chnls, out_chnls, stride=1, dnsmpl=None):
        super(ResBlk, self).__init__()
        #
        self.conv1 = nn.Conv2d(in_chnls, out_chnls, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chnls)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_chnls, out_chnls, 3, stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chnls)
        self.dnsmpl = dnsmpl


    def forward(self, input):
        residual = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.dnsmpl:
            residual = self.dnsmpl(input)
        x = x + residual
        output = self.relu(x)
        return output


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_cls):
        super(ResNet, self).__init__()
        #
        self.in_chnls = 16
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.fc = nn.Linear(64, num_cls)

    def make_layer(self, block, out_chnls, blocks, stride=1):
        dnsmpl = None
        if (stride != 1) or (self.in_chnls != out_chnls):
            dnsmpl = nn.Sequential(
                nn.Conv2d(self.in_chnls, out_chnls, 3, stride=stride * blocks, padding=1),
                nn.BatchNorm2d(out_chnls)
            )
        layers = []
        layers.append(block(self.in_chnls, out_chnls, stride, dnsmpl))
        self.in_chnls = out_chnls
        for i in range(1, blocks):
            layers.append(block(out_chnls, out_chnls))
        return nn.Sequential(*layers)

    def forward(self, iput):
        x = self.conv(iput)
        x = self.bn(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        #
        self.fetr = x
        #
        x = self.fc(x)
        return x


model = ResNet(ResBlk, [2, 2, 2], 10).to(device)
model.load_state_dict(torch.load('.\\parm(temp).pkl'))

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# for updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# train model
pre_epoch_total_step = len(trainloader)
current_lr = learning_rate
for epoch in range(EPOCHS):
    for i, (x, y) in enumerate(trainloader):
        x = x.to(device)
        y = y.to(device)

        # forward
        prediction = model(x)
        loss = criterion(prediction, y)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 5 == 0:
            template = r"Epoch:{}/{}, step:{}/{}, Loss:{:.6f}"
            print(template.format(epoch + 1, EPOCHS, i + 1, pre_epoch_total_step, loss.item()))

    # decay learning rate
    if (epoch + 1) % 20 == 0:
        current_lr = current_lr * 0.9
        update_lr(optimizer, current_lr)
        torch.save(model.state_dict(), '.\\parm.pkl')

# test model
model.eval()
with torch.no_grad():
    total = 0
    correct = 0
    for x, y in testloader:
        x = x.to(device)
        y = y.to(device)
        prediction = model(x)
        _, predic = torch.max(prediction.data, dim=1)
        total += y.shape[0]
        correct += (predic == y).sum().item()

    print("Accuracy:{}%".format(100 * correct / total))
