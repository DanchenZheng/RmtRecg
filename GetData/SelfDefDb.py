#
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from PIL import Image
import torchvision.transforms as tvxfrm
from torch.utils.data import Dataset
#
from GetData.AsstRela import FolderEnum
from GetData.PrepRela import PrepRela


# Self Defined Dataset Class
class SelfDefDb(Dataset):
    def __init__(self, root_pth, fl_flg: FolderEnum, fmt_xfrm=tvxfrm.ToTensor(), augment=None):
        self.root_dir: str = root_pth
        self.fl_str = fl_flg.value
        self.aug = augment
        self.xfrm = fmt_xfrm
        #
        self.DataDict, self.LabelDict = PrepRela.gnrtloc(root_pth, fl_flg.get_fllst())
        #
        self.smpl_num = len(self.LabelDict[self.fl_str])
        #
        img, _ = self.__getitem__(0)
        self.img_size = img.size()

    def __len__(self):
        return self.smpl_num

    def __imgsize(self):
        return self.img_size

    def __getitem__(self, idx):
        #
        if self.__len__() > idx >= 0:
            #
            img_pth = self.DataDict[self.fl_str][idx]
            img_lbl = self.LabelDict[self.fl_str][idx]
            #
            if not os.path.isfile(img_pth):
                print('\"' + img_pth + '\" does not exist!')
                return None, None
            #
            return self.__rd_cvrt_img(img_pth), img_lbl
        else:
            return None, None

    def __rd_cvrt_img(self, img_pth):
        img = Image.open(img_pth)
        #
        if self.aug:
            img = self.aug(img)
        #
        if self.xfrm:
            img = self.xfrm(img).float()
        #
        return img


# Resize for Input Image
class Resize(object):
    def __init__(self, oput_size: tuple):
        self.oput_size = oput_size

    def __call__(self, img):
        return tvxfrm.resize(img, self.output_size)


# Tensor on Image
class ToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.transpose(img, (2, 0, 1)))


# Display Images form List
class DplySmplImg(object):
    def __init__(self, root_pth, dply_lst=None):
        #
        # tmp_db = SelfDefDb(root_path, gv_xfrm=tvxfrm.Compose([Resize((50, 50)), ToTensor()]))
        tmp_db = SelfDefDb(root_pth, xfrm=None)
        #
        if not dply_lst:
            dply_lst = random.sample(range(1, tmp_db.__len__()), 16)
        #
        plt.figure()
        #
        grid_num = np.ceil(np.sqrt(len(dply_lst)))
        #
        for idx in range(len(dply_lst)):
            ele = tmp_db.__getitem__(dply_lst[idx])
            ax = plt.subplot(grid_num, grid_num, idx + 1)
            plt.imshow(ele["image"])
            ax.set_title('label {}'.format(ele["label"]))
            ax.axis('off')
        #
        plt.show()


def exec_usecase():
    """
    Test related methods or classes
    2020-9-17
    """
    #
    # DplySmplImg("F:/Remote/RemoteData/")
    # tmp_db = SelfDefDb("F:/Remote/RemoteData/")
    #
    # a, b = tmp_db.__getitem__(1000)

    a = SelfDefDb("F:/Remote/RemoteData/", FolderEnum.TRAIN, 0)
    print("Preparaion of Data Is Finished!")
