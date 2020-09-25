#
import os
import random
import shutil
import numpy as np
#
from collections import Counter
from os.path import isdir, join
from pathlib import Path
from enum import Enum


# Enum of train/test
class FolderEnum(Enum):
    TRAIN = "train"
    TEST = "test"

    @staticmethod
    def get_fllst():
        #
        fllst = []
        #
        for tmp in FolderEnum:
            fllst.append(tmp.value)
        #
        return fllst


# Enum of file types, e.g. image
class FileTypeEnum(Enum):
    IMAGE = [".jpg", ".png"]


class AsstRela(object):
    """
    Static Class with Methods for
    2020-9-9
    """

    @staticmethod
    def chkrept(iput_lst: list) -> bool:
        """
        Check whether exists repeated elements
        2020-9-9
        """
        #
        iput_dict = dict(Counter(iput_lst))
        #
        for key in iput_dict:
            if iput_dict[key] > 1:
                return True
        #
        return False

    @staticmethod
    def cntnum(root_pth, file_type: FileTypeEnum) -> int:
        """
        Count the number of files in the given folder
        2020-9-9
        """
        #
        img_num = 0
        #
        for ele_file in os.scandir(root_pth):
            if AsstRela.checkfile(ele_file.name, file_type):
                img_num = img_num + 1
            #
        return img_num

    @staticmethod
    def chkfile(img_name, file_type: FileTypeEnum) -> bool:
        """
        Check whether the input file is the given file_type
        2020-9-9
        """
        #
        for endStr in file_type.value:
            if img_name.endswith(endStr):
                return True
        #
        return False

    @staticmethod
    def rden_dict(dst_dict: dict) -> dict:
        """
        Reduce common divisors from the elements of the list
        2020-9-9
        """
        # Start from the smallest value
        div_val = 2
        # The maximum  possible div_val
        max_val: int = np.floor(max(dst_dict.values()) / 2)
        #
        while div_val <= max_val:
            # Elements are divisible by div_val
            div_flg = True
            #
            for tmpVal in dst_dict.values():
                #
                if not tmpVal % div_val == 0:
                    div_val = div_val + 1
                    div_flg = False
                    break
            #
            if div_flg:
                break
        #
        if div_val <= max_val:
            #
            for key, val in dst_dict.items():
                dst_dict[key] = int(val / div_val)
            #
            return AsstRela.rden_dict(dst_dict)
        else:
            return dst_dict

    @staticmethod
    def cacdir(dst_dict: dict, src_pth, ele_type: FileTypeEnum):
        """
        Check and copy elements with reduce ratio list
        2020-9-9
        """
        AsstRela.__cacdir__(AsstRela.rden_dict(dst_dict), src_pth, ele_type)
        #
        print("Copy Finished!")
        #
        return

    @staticmethod
    def __rndmrden__(dst_dict: dict) -> list:
        """
        Generate list with random indexes
        2020-9-9
        """
        #
        tmp_lst = []
        for key, val in dst_dict.items():
            tmp_lst.extend([key] * val)
        #
        random.shuffle(tmp_lst)
        #
        return tmp_lst

    @staticmethod
    def __cacdir__(dst_dict: dict, src_pth, ele_type: FileTypeEnum, pfx_flg: bool = False):
        """
        Check and copy folders, check and copy elements according to ratio, recursion method
        2020-9-9    dstpth_lst: list, rden_lst: list
        """
        # Check whether reference path exists
        if not os.path.exists(src_pth):
            raise ValueError("Reference Path Does Not Exits!")
        #
        dict_len = len(dst_dict)
        copy_idx = 0
        # Check reference root directory
        for src_sub in os.scandir(src_pth):
            # Check subdirectory
            if isdir(src_sub):
                dstsub_dict: dict = {}
                #
                for key, val in dst_dict.items():
                    dst_sub = join(key, src_sub.name)
                    dstsub_dict[dst_sub] = val
                    if not os.path.exists(dst_sub):
                        os.makedirs(dst_sub)
                #
                AsstRela.__cacdir__(dstsub_dict, src_sub, ele_type)
                #
                print("'" + src_sub.name + "' Copy Finished!")
            #
            if AsstRela.chkfile(src_sub.name, ele_type):
                #
                if copy_idx == 0:
                    dst_lst = AsstRela.__rndmrden__(dst_dict)
                #
                shutil.copyfile(src_sub.path, AsstRela.__copy2path__(dst_lst[copy_idx % dict_len], src_sub.name))
                #
                copy_idx = copy_idx + 1

    @staticmethod
    def __copy2path__(dst_path, file_name) -> str:
        #
        tmpdst_path = join(dst_path, file_name)
        #
        if Path(tmpdst_path).exists():
            tmplst = file_name.split('_')
            fnl_str = tmplst[-1]
            #
            if tmplst[-1].isalnum():
                return AsstRela.__copy2path__(dst_path, file_name[:-len(fnl_str)] + str(int(tmplst[-1]) + 1))
            else:
                return join(dst_path, file_name + '_1')
        else:
            return tmpdst_path


def exec_usecase():
    """
    Test related methods or classes
    2020-9-17
    """
    print("Preparation of Data Is Finished!")
    # #
    # trgt_lst = ["F:\\Tracing Scale\\FavData\\train",
    #             "F:\\Tracing Scale\\FavData\\test"]
    # #
    # radio_lst = [4, 1]
    #
    ref_path = "F:\\Tracing Scale\\InitFavData"
    #
    trgt_dict = {"F:\\Tracing Scale\\FavData\\train": 40, "F:\\Tracing Scale\\FavData\\test": 10}
    #
    AsstRela.cacdir(trgt_dict, ref_path, FileTypeEnum.IMAGE)
    #
    print("Preparation of Data Is Finished!")

# exec_usecase()