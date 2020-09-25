#
import os
from os.path import isdir, join
#
from GetData.AsstRela import FileTypeEnum, AsstRela, FolderEnum


class PrepRela(object):
    @staticmethod
    def div_tatdata(src_pth: str, dst_pth: str, folder_dict: dict,
                    ele_type: FileTypeEnum):
        """
        Divide source dataset into Train Dataset and Test Dataset
        2020-9-11
        """
        # Check original path
        if not isdir(src_pth):
            raise ValueError("Incorrect Source Folder Path!")
        # Check root path
        if not isdir(dst_pth):
            raise ValueError("Incorrect Destination Folder Path!")
        # Check ratio
        for val in folder_dict.values():
            if not isinstance(val, int):
                raise ValueError("Value of 'ratio' Is Not Integer!")
        #
        par_dic = {}
        #
        for key, val in folder_dict.items():
            # Check Folder
            ele_pth = join(dst_pth, key)
            #
            if not os.path.exists(ele_pth):
                os.makedirs(ele_pth)
            #
            for tmp in os.scandir(ele_pth):
                raise ValueError("Folder of '" + key + "' Is Not Empty!")
            #
            par_dic[ele_pth] = val
        #
        AsstRela.cacdir(par_dic, src_pth, ele_type)
        #
        return

    @staticmethod
    def merg_tatdata(src_pth: str, dst_pth: str, fls_lst: list,
                     ele_type: FileTypeEnum):
        """
        Merge Train Dataset and Test Dataset into one dataset
        2020-9-21
        """
        # Check original path
        if not isdir(src_pth):
            raise ValueError("Incorrect Source Folder Path!")
        # Check root path
        if not isdir(dst_pth):
            raise ValueError("Incorrect Destination Folder Path!")
        #
        for tmp in os.scandir(dst_pth):
            raise ValueError("Folder of '" + tmp + "' Is Not Empty!")
        #
        for flnm in fls_lst:
            ele_pth = join(src_pth, flnm)
            AsstRela.cacdir({dst_pth: 1}, ele_pth, ele_type)
        #
        return

    @staticmethod
    def gnrtloc(root_pth: str, fls_lst: list, txt_flg: bool = False,
                pfx_str: str = ""):
        """
        Prepare 'dict' and 'txt' according to the Train/Test Dataset in root path
        2020-9-11
        """
        # Check root path
        if not isdir(root_pth):
            raise ValueError("Incorrect Root File Path")
        #
        lastdir_lst: list = None
        # Generate dict with lists
        for flnm in fls_lst:
            #
            tmp_pth = join(root_pth, flnm)
            #
            curdir_lst = [ele_dir for ele_dir in os.scandir(tmp_pth) if isdir(ele_dir.path)]
            #
            if lastdir_lst:
                for idx in range(len(lastdir_lst)):
                    if len(curdir_lst) < idx or not curdir_lst[idx] == curdir_lst[idx]:
                        raise ValueError("Folders Do Not Match!")
            else:
                lastdir_lst = curdir_lst
        #
        data_dict: dict[str, list] = dict()
        label_dict: dict[str, list] = dict()
        #
        for flnm in fls_lst:
            #
            data_dict[flnm] = []
            label_dict[flnm] = []
            #
            wrt_str = ""
            whl_pfx = pfx_str + "\\" + flnm + "\\"
            subdir_lst: list = [ele_dir for ele_dir in os.scandir(join(root_pth, flnm)) if
                                isdir(ele_dir.path)]
            #
            for idx in range(len(subdir_lst)):
                #
                img_lst: list = [imgfl for imgfl in os.scandir(subdir_lst[idx].path)
                                 if AsstRela.chkfile(imgfl.name, FileTypeEnum.IMAGE)]
                #
                for img_ele in img_lst:
                    wrt_str = wrt_str + whl_pfx + subdir_lst[idx].name + "\\" + img_ele.name + " " + str(
                        idx) + "\n"
                    #
                    data_dict[flnm].append(img_ele.path)
                    label_dict[flnm].append(idx)
            #
            if txt_flg:
                txtfile = open(root_pth + flnm + ".txt", "w")
                txtfile.write(wrt_str)
                txtfile.close
        #
        return data_dict, label_dict


def exec_usecase():
    """
    Test related methods or classes
    2020-9-17
    """
    # print("Preparation of Data Is Finished!")
    # dataDict, labelDict = PrepRela.gnrtloc("F:\\Remote\\RemoteData\\",
    #                                        FolderEnum.get_fllst(), True)
    # print("Preparation of Data Is Finished!")

    # #
    # folder_dict = {FolderEnum.TRAIN.value: 4, FolderEnum.TEST.value: 1}
    # #
    # PrepRela.div_tatdata("F:\\Tracing Scale\\InitFavData",
    #                      "F:\\Tracing Scale\\FavData", folder_dict, FileTypeEnum.IMAGE)
    #
    PrepRela.merg_tatdata("F:\\Remote\\RemoteData", "F:\\Remote\\tmp", ['train', 'test'], FileTypeEnum.IMAGE)
    #
    print("Data Division Is Finished!")


exec_usecase()
