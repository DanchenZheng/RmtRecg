#
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as xfrm

# device configuration
from torch.utils.data import Dataset, DataLoader
from GetData.AsstRela import FolderEnum
from GetData.SelfDefDb import SelfDefDb

dvc = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# hyper parameters
EPOCHS = 20
batch_size = 512
learning_rate = 0.01
classes = 10
wah = 64

# Image preprocessing modules
trn_xfrm = xfrm.Compose([
    xfrm.Grayscale(),
    xfrm.Resize(wah),
    xfrm.Pad(4),
    xfrm.RandomCrop(wah),
    xfrm.ToTensor(),
])

test_xfrm = xfrm.Compose([
    xfrm.Grayscale(),
    xfrm.Resize(wah),
    xfrm.ToTensor(),
])

# 训练集加载
trn_data = SelfDefDb("F:\\Remote\\RemoteData", FolderEnum.TRAIN, fmt_xfrm=trn_xfrm)
trn_loader = DataLoader(dataset=trn_data, batch_size=batch_size, shuffle=True)
# 测试集加载
test_data = SelfDefDb("F:\\Remote\\RemoteData", FolderEnum.TEST, fmt_xfrm=test_xfrm)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)


#
class ResBlk(nn.Module):
    """
    Block for ResNet
    """

    def __init__(self, in_chnls, out_chnls, strd=1, dnsmpl=None):
        super(ResBlk, self).__init__()
        #
        self.conv1 = nn.Conv2d(in_chnls, out_chnls, 3, strd, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chnls)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_chnls, out_chnls, 3, strd, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chnls)
        self.dnsmpl = dnsmpl

    def forward(self, iput):
        residual = iput
        x = self.conv1(iput)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.dnsmpl:
            residual = self.dnsmpl(iput)
        x = x + residual
        oput = self.relu(x)
        return oput


# ResNet
class ResNet(nn.Module):
    """
    Self defined ResNet
    """

    def __init__(self, block, lynum_lst, num_cls):
        super(ResNet, self).__init__()
        #
        self.in_chnls = 16
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        #
        self.layer1 = self.__make_layer(block, 16, lynum_lst[0])
        self.layer2 = self.__make_layer(block, 32, lynum_lst[1], 2)
        self.layer3 = self.__make_layer(block, 64, lynum_lst[2], 2)
        self.avg_pool = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Linear(64, num_cls)

    def __make_layer(self, block, out_chnls, blk_num, stride=1):
        dnsmpl = None
        if (stride != 1) or (self.in_chnls != out_chnls):
            dnsmpl = nn.Sequential(
                nn.Conv2d(self.in_chnls, out_chnls, 3, stride=stride * blk_num, padding=1),
                nn.BatchNorm2d(out_chnls)
            )
        layers = [block(self.in_chnls, out_chnls, stride, dnsmpl)]
        self.in_chnls = out_chnls
        for i in range(1, blk_num):
            layers.append(block(out_chnls, out_chnls))
        return nn.Sequential(*layers)

    def forward(self, iput):
        x = self.conv(iput)
        x = self.bn(x)
        x = self.relu(x)
        #
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


model = ResNet(ResBlk, [2, 2, 2], 6).to(dvc)
model.load_state_dict(torch.load('.\\parm.pkl'))

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# for updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# # train model
# pre_epoch_total_step = len(trn_loader)
# current_lr = learning_rate
# for epoch in range(EPOCHS):
#     for i, (x, y) in enumerate(trn_loader):
#         x = x.to(dvc)
#         y = y.to(dvc)
#
#         # forward
#         prediction = model(x)
#         loss = criterion(prediction, y)
#
#         # backward
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if (i + 1) % 5 == 0:
#             template = r"Epoch:{}/{}, step:{}/{}, Loss:{:.6f}, "
#             print(template.format(epoch + 1, EPOCHS, i + 1, pre_epoch_total_step, loss.item()))
#
#     # decay learning rate
#     if (epoch + 1) % 10 == 0:
#         current_lr = current_lr * 0.9
#         update_lr(optimizer, current_lr)
#         torch.save(model.state_dict(), '.\\parm.pkl')

# test model
model.eval()
with torch.no_grad():
    total = 0
    correct = 0
    for x, y in test_loader:
        x = x.to(dvc)
        y = y.to(dvc)
        prediction = model(x)
        _, predic = torch.max(prediction.data, dim=1)
        total += y.shape[0]
        correct += (predic == y).sum().item()

    print("Accuracy:{}%".format(100 * correct / total))
