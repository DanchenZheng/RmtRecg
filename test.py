#
# from GetData.WhlTesDb import WhlDataset
import Backup.WhlRmtDb as RmtDb

#
rmtTrain_Path = "F:/Remote/RemoteData/train"
#

if __name__ == '__main__':
    tmpDataset = RmtDb.LoadTraindata(rmtTrain_Path)
    a = 2



# #
# plt.figure()
# #
# strtIdx = 15000
# #
# for idx in range(strtIdx, strtIdx+12):
#     plt.subplot(3, 4, idx+1-strtIdx)
#     plt.imshow(tmpDataset[idx].numpy().transpose(1, 2, 0))
# #
# plt.show()