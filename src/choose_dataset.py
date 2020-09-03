# implemented by JunfengHu
# create time: 7/20/2019
import numpy as np
import torch
import scipy.io as sio
import load_data as loader


class DatasetChooser(object):

    def __init__(self, config):
        self.config = config
        self.dataset = config.dataset

    def choose_dataset(self, train=True, prediction=False):

        if not prediction:
            if self.config.datatype == 'lie':
                if self.dataset == 'Human':
                    bone_length_path = None
                    data = loader.HumanDataset(self.config, train=train)
                    self.config.input_size = data[0]['encoder_inputs'].shape[1]
                elif self.dataset == 'Mouse':
                    bone_length_path = './data/Mouse/Test/y_test_lie/test_0_lie.mat'
                    data = loader.MouseDataset(self.config, train=train)
                    self.config.input_size = data[0]['encoder_inputs'].shape[1]
            elif self.config.datatype == 'smpl':
                bone_length_path = None
                data = loader.CMUDataset(self.config, train=train)
                self.config.input_size = data[0]['encoder_inputs'].shape[1]

        else:
            if self.config.datatype == 'lie':
                if self.dataset == 'Human':
                    bone_length_path = None
                    data_loader = loader.HumanPredictionDataset(self.config)
                    data = data_loader.get_data()
                    self.config.input_size = data[0][list(data[0].keys())[0]].shape[2]
                elif self.dataset == 'Mouse':
                    bone_length_path = './data/Mouse/Test/y_test_lie/test_0_lie.mat'
                    data_loader = loader.AnimalPredictionDataset(self.config)
                    data = data_loader.get_data()
                    self.config.input_size = data[0][list(data[0].keys())[0]].shape[2]
            elif self.config.datatype == 'smpl':
                bone_length_path = None
                data_loader = loader.CMUPredictionDataset(self.config)
                data = data_loader.get_data()

        if bone_length_path is not None:
            rawdata = sio.loadmat(bone_length_path)
            rawdata = rawdata[list(rawdata.keys())[3]]
            bone = self.cal_bone_length(rawdata)
        else:
            # AMASS的骨骼长度计算smpl中性人关节坐标得到，# 共21行，非零行19(左右脚趾是零）
            bone = np.array([[122.1, 0., 0.],  # 1
                             [120., 0., 0.],  # 2
                             [121.9, 0., 0.],  # 3
                             [380.8, 0., 0.], # 4
                             [391.6, 0., 0.], # 5
                             [139., 0., 0.],  # 6
                             [408.2, 0., 0.], # 7
                             [410.7, 0., 0.], # 8
                             [61., 0., 0.],  # 9
                             [0., 0., 0.],   # 10
                             [0., 0., 0.],   # 11
                             [220.5, 0., 0.], # 12
                             [153., 0., 0.],  # 13
                             [154.3, 0., 0.],  # 14
                             [82.8, 0., 0.],  # 15
                             [99.4, 0., 0.],  # 16
                             [107.9, 0., 0.],  # 17
                             [260.7, 0., 0.], # 18
                             [257.6, 0., 0.], # 19
                             [247., 0., 0.], # 20
                             [256.3, 0., 0.]])# 21


            # bone = np.array([[0., 0., 0.], # 共32行，非零行26
            #                  [132.95, 0., 0.],
            #                  [442.89, 0., 0.],
            #                  [454.21, 0., 0.],
            #                  [162.77, 0., 0.],
            #                  [75., 0., 0.],
            #                  [132.95, 0., 0.],
            #                  [442.89, 0., 0.],
            #                  [454.21, 0., 0.],
            #                  [162.77, 0., 0.],
            #                  [75., 0., 0.],
            #                  [0., 0., 0.],
            #                  [233.38, 0., 0.],
            #                  [257.08, 0., 0.],
            #                  [121.13, 0., 0.],
            #                  [115., 0., 0.],
            #                  [257.08, 0., 0.],
            #                  [151.03, 0., 0.],
            #                  [278.88, 0., 0.],
            #                  [251.73, 0., 0.],
            #                  [0., 0., 0.],
            #                  [100., 0., 0.],
            #                  [137.5, 0., 0.],
            #                  [0., 0., 0.],
            #                  [257.08, 0., 0.],
            #                  [151.03, 0., 0.],
            #                  [278.88, 0., 0.],
            #                  [251.73, 0., 0.],
            #                  [0., 0., 0.],
            #                  [100., 0., 0.],
            #                  [137.5, 0., 0.],
            #                  [0., 0., 0.]])

        return data, bone

    def __call__(self, train=True, prediction=False):
        return self.choose_dataset(train, prediction)

    def cal_bone_length(self, rawdata):

        njoints = rawdata.shape[1]
        bone = np.zeros([njoints, 3])
        if self.config.datatype == 'lie':
            for i in range(njoints):
                bone[i, 0] = round(rawdata[0, i, 3], 2)
            # delete zero in bone, n joints mean n-1 bones
            bone = bone[1:, :]

        elif self.config.datatype == 'xyz':
            for i in range(njoints):
                bone[i, 0] = round(np.linalg.norm(rawdata[0, i, :] - rawdata[0, i - 1, :]), 2)

        return bone

