# implemented by JunfengHu
# create time: 7/20/2019
from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import copy
import utils
import sys
import os


class FormatDataPre(object):
    """
    Form prediction(test) data.
    """

    def __init__(self):
        pass

    def __call__(self, x_test, y_test):
        dec_in_test = x_test[-1:, :]
        x_test = x_test[:-1, :]
        return {'x_test': x_test, 'dec_in_test': dec_in_test, 'y_test': y_test}


class FormatData(object):
    """
    Form train/validation data.
    """

    def __init__(self, config):
        self.config = config

    def __call__(self, sample, train):

        total_frames = self.config.input_window_size + self.config.output_window_size
        # CMU sample [375,60]
        video_frames = sample.shape[0]
        idx = np.random.randint(1, video_frames - total_frames) #在可选范围中随机挑选帧起始点

        data_seq = sample[idx:idx + total_frames, :]
        encoder_inputs = data_seq[:self.config.input_window_size - 1, :]
        # 最后一个弃掉了,这里代码还可以精简
        if train:
            decoder_inputs = data_seq[self.config.input_window_size - 1:
                                      self.config.input_window_size - 1 + self.config.output_window_size, :]
        else:
            decoder_inputs = data_seq[self.config.input_window_size - 1:self.config.input_window_size, :]
        decoder_outputs = data_seq[self.config.input_window_size:, :]
        return {'encoder_inputs': encoder_inputs, 'decoder_inputs': decoder_inputs, 'decoder_outputs': decoder_outputs}


class LieTsfm(object):
    """
    This class is redundant and could be integrated into dataset class. However, we didn't do that due to some historical events.
    """

    def __init__(self, config):
        self.config = config

    def __call__(self, sample):
        rawdata = sample

        data = rawdata[:, :-1, :3].reshape(rawdata.shape[0], -1)
        return data

class CMUDataset(Dataset):

    def __init__(self, config, train=True):

        self.config = config
        self.train = train
        self.formatdata = FormatData(config)
        if config.datatype == 'smpl':
            train_path = config.data_root
        else:
            print('CMUDataset only support the smpl datatype')
            sys.exit(1)
        if config.filename != 'all' :
            if train:
                subjects = config.subjects_train
            else:
                subjects = config.subjects_test
        else:
            print('Only support walking and dance action')
            sys.exit(1)

        set = []
        complete_train = []
        for sub in subjects:
            folderdir = '{0}/{1}'.format(train_path, sub)
            for file in os.listdir(folderdir):
                filedir = '{0}/{1}'.format(folderdir, file)
                rawdata = np.load(filedir)['poses'][:, :66]
                rawdata = self.frame_filter(rawdata)
                if rawdata.shape[0] > 70:
                    set.append(rawdata)
            if len(complete_train) == 0:
                complete_train = copy.deepcopy(set[-1]) #每个subjects取最后一个动作序列计算均值方差
            else:
                complete_train = np.append(complete_train, set[-1], axis=0)
        print('视频个数：',len(set))
        if not train and config.data_mean is None:
            print('Load train dataset first!')
        if train and config.datatype == 'smpl':
            data_mean, data_std, dim_to_ignore, dim_to_use = utils.normalization_stats(complete_train)
            config.data_mean = data_mean
            config.data_std = data_std
            config.dim_to_ignore = dim_to_ignore
            config.dim_to_use = dim_to_use

        set = utils.normalize_data(set, config.data_mean, config.data_std, config.dim_to_use)
        # [S_num, frame_for_S, 66]
        self.data = set
    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        sample = self.formatdata(self.data[idx], False)
        return sample
    def frame_filter(self, rawdata):
        '''
        author: zhouhonghong
        过滤掉前后的静止画面
        :return:
        '''
        print('using frame_filter!!!')
        forward_frame = rawdata[0, :]
        remain_id = []
        for id in range(rawdata.shape[0] - 1):
            this_frame = rawdata[id + 1, :]
            if np.sum(np.abs(this_frame - forward_frame)) > 0.1:
                remain_id.append(id + 1)
            forward_frame = this_frame
        start_id = remain_id[0]
        end_id = remain_id[-1]
        if abs(start_id-end_id) > 30:
            return rawdata[start_id:end_id-30, :]
        else:
            return rawdata[start_id, :]



class HumanDataset(Dataset):

    def __init__(self, config, train=True):

        self.config = config
        self.train = train
        self.lie_tsfm = LieTsfm(config)
        self.formatdata = FormatData(config)
        if config.datatype == 'lie':
            if train:
                train_path = './data/h3.6m/Train/train_lie'
            else:
                train_path = './data/h3.6m/Test/test_lie'
        elif config.datatype == '':
            train_path = './data/h3.6m/Train/train_xyz'
        if train:
            subjects = ['S1', 'S6', 'S7', 'S8', 'S9', 'S11']
        else:
            subjects = ['S5']

        if config.filename == 'all':
            actions = ['directions', 'discussion', 'eating', 'greeting', 'phoning', 'posing', 'purchases', 'sitting',
                       'sittingdown', 'smoking', 'takingphoto', 'waiting', 'walking', 'walkingdog', 'walkingtogether']
        else:
            actions = [config.filename]

        set = []
        complete_train = []
        for id in subjects:
            for action in actions:
                for i in range(2):
                    if config.datatype == 'lie':
                        filename = '{0}/{1}_{2}_{3}_lie.mat'.format(train_path, id, action, i + 1)
                        rawdata = sio.loadmat(filename)['lie_parameters']
                        set.append(rawdata)
                    elif config.datatype == 'xyz':
                        filename = '{0}/{1}_{2}_{3}_xyz.mat'.format(train_path, id, action, i + 1)
                        rawdata = sio.loadmat(filename)['joint_xyz']
                        set.append(rawdata.reshape(rawdata.shape[0], -1))

                if len(complete_train) == 0:
                    complete_train = copy.deepcopy(set[-1])
                else:
                    complete_train = np.append(complete_train, set[-1], axis=0)

        if not train and config.data_mean is None:
            print('Load train dataset first!')

        if train and config.datatype == 'lie':
            data_mean, data_std, dim_to_ignore, dim_to_use = utils.normalization_stats(complete_train)
            config.data_mean = data_mean
            config.data_std = data_std
            config.dim_to_ignore = dim_to_ignore
            config.dim_to_use = dim_to_use

        set = utils.normalize_data(set, config.data_mean, config.data_std, config.dim_to_use)
        # [S_num, frame_for_S, 54]
        self.data = set

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        if self.config.datatype == 'lie':
            pass
        elif self.config.datatype == 'xyz':
            pass
        sample = self.formatdata(self.data[idx], False)
        return sample


class MouseDataset(Dataset):

    def __init__(self, config, train):
        self.config = config
        self.train = train
        self.lie_tsfm = LieTsfm(config)
        self.formatdata = FormatData(config)
        if config.datatype == 'lie':
            train_path = './data/Mouse/Train/train_lie/'
            tail = '_lie.mat'
        elif config.datatype == 'xyz':
            train_path = './data/Mouse/Train/train_xyz/'
            tail = '_xyz.mat'
        if train:
            subjects = ['S1', 'S3', 'S4']
        else:
            subjects = ['S2']

        set = []
        for id in subjects:
            filename = train_path + id + tail
            rawdata = sio.loadmat(filename)
            rawdata = rawdata[list(rawdata.keys())[3]]
            set.append(rawdata)
        self.data = set

    def __getitem__(self, idx):

        if self.config.datatype == 'lie':
            sample = self.data[idx][:, :-1, :3].reshape(self.data[idx].shape[0], -1)

            # sample = self.lie_tsfm(self.data[idx])
        elif self.config.datatype == 'xyz':
            pass
        sample = self.formatdata(sample, False)
        return sample

    def __len__(self):

        return len(self.data)


class AnimalPredictionDataset(object):

    def __init__(self, config):
        self.config = config

        if config.datatype == 'lie':
            x = []
            y = []
            if self.config.dataset == 'Mouse':
                set_name = 'Mouse'
            for i in range(8):
                x_filename = './data/' + set_name + '/Test/x_test_lie/test_' + str(i) + '_lie.mat'
                y_filename = './data/' + set_name + '/Test/y_test_lie/test_' + str(i) + '_lie.mat'

                x_rawdata = sio.loadmat(x_filename)
                x_rawdata = x_rawdata[list(x_rawdata.keys())[3]]

                y_rawdata = sio.loadmat(y_filename)
                y_rawdata = y_rawdata[list(y_rawdata.keys())[3]]

                x_data = x_rawdata[:, :-1, :3].reshape(x_rawdata.shape[0], -1)
                x.append(x_data)

                y_data = y_rawdata[:, :-1, :3].reshape(y_rawdata.shape[0], -1)
                y.append(y_data)

        x = np.array(x)
        y = np.array(y)
        dec_in_test = np.reshape(x[:, -1, :], [x.shape[0], 1, x.shape[2]])
        x = x[:, 0:-1, :]

        self.x_test_dict = {}
        self.y_test_dict = {}
        self.dec_in_test_dict = {}

        self.x_test_dict['default'] = x
        self.y_test_dict['default'] = y
        self.dec_in_test_dict['default'] = dec_in_test

    def get_data(self):

        return [self.x_test_dict, self.y_test_dict, self.dec_in_test_dict]

class CMUPredictionDataset(object):

    def __init__(self, config):
        self.config = config
        self.action = config.filename
        test_set = {}
        self.file_names = [[],] # 列表，顺序记录各subject文件夹下的文件名称
        for sub in config.subjects_test:
            folderdir = '{0}/{1}'.format(config.data_root, sub)
            for filename in os.listdir(folderdir):
                filedir = '{0}/{1}'.format(folderdir, filename)
                test_set[(sub, filename)] = np.load(filedir)['poses'][:, :66]

        try:
            config.data_mean
        except NameError:
            print('Load  train set first!')

        self.test_set = utils.normalize_data_dir(test_set, config.data_mean, config.data_std, config.dim_to_use)

    def get_data(self):
        x_test = {}
        y_test = {}
        dec_in_test = {}
        encoder_inputs, decoder_inputs, decoder_outputs = self.get_batch_srnn(self.config, self.test_set,
                                                                                  self.config.output_window_size)
        x_test[self.action] = encoder_inputs
        y_test[self.action] = decoder_outputs
        dec_in_test[self.action] = np.zeros([decoder_inputs.shape[0], 1, decoder_inputs.shape[2]])
        dec_in_test[self.action][:, 0, :] = decoder_inputs[:, 0, :]
        return [x_test, y_test, dec_in_test]

    def get_batch_srnn(self, config, data, target_seq_len):
        # Obtain SRNN test sequences using the specified random seeds

        frames = {}
        frames[self.action] = self.find_indices_srnn(data)

        batch_size = 5 ##  测试集视频数目
        source_seq_len = config.input_window_size

        seeds = [(frames[self.action][i]) for i in range(batch_size)]

        encoder_inputs = np.zeros((batch_size, source_seq_len - 1, config.input_size), dtype=float)
        decoder_inputs = np.zeros((batch_size, target_seq_len, config.input_size), dtype=float)
        decoder_outputs = np.zeros((batch_size, target_seq_len, config.input_size), dtype=float)

        for i in range(batch_size):
            idx = seeds[i]
            idx = idx +20

            data_sel = data[self.keys[i]]

            data_sel = data_sel[(idx - source_seq_len):(idx + target_seq_len), :]

            encoder_inputs[i, :, :] = data_sel[0:source_seq_len - 1, :]  # x_test
            decoder_inputs[i, :, :] = data_sel[source_seq_len - 1:(source_seq_len + target_seq_len - 1), :]  # decoder_in_test
            decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]  # y_test

        return [encoder_inputs, decoder_inputs, decoder_outputs]

    def find_indices_srnn(self, data):

        """
        Obtain the same action indices as in SRNN using a fixed random seed
        See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py
        """

        SEED = 1234567890
        rng = np.random.RandomState(SEED)
        prefix, suffix = 20, 50

        idx = []
        self.keys = []
        for key in data.keys():
            idx.append(rng.randint(1, data[key].shape[0] - prefix - suffix)) #前100帧和后150帧是静止的，不要
            self.keys.append(key)

        return idx

class HumanPredictionDataset(object):

    def __init__(self, config):
        self.config = config
        if config.filename == 'all':
            self.actions = ['directions', 'discussion', 'eating', 'greeting', 'phoning', 'posing', 'purchases',
                            'sitting', 'sittingdown', 'smoking', 'takingphoto', 'waiting', 'walking', 'walkingdog',
                            'walkingtogether']
        else:
            self.actions = [config.filename]

        test_set = {}
        for subj in [5]:
            for action in self.actions:
                for subact in [1, 2]:
                    if config.datatype == 'lie':
                        filename = '{0}/S{1}_{2}_{3}_lie.mat'.format('./data/h3.6m/Test/test_lie', subj, action, subact)
                        test_set[(subj, action, subact)] = sio.loadmat(filename)['lie_parameters']

                    if config.datatype == 'xyz':
                        filename = '{0}/S{1}_{2}_{3}_xyz.mat'.format('./data/h3.6m/Test/test_xyz', subj, action, subact)
                        test_set[(subj, action, subact)] = sio.loadmat(filename)['joint_xyz']
                        test_set[(subj, action, subact)] = test_set[(subj, action, subact)].reshape(
                            test_set[(subj, action, subact)].shape[0], -1)
        try:
            config.data_mean
        except NameError:
            print('Load  train set first!')
        self.test_set = utils.normalize_data_dir(test_set, config.data_mean, config.data_std, config.dim_to_use)

    def get_data(self):
        x_test = {}
        y_test = {}
        dec_in_test = {}
        for action in self.actions:
            encoder_inputs, decoder_inputs, decoder_outputs = self.get_batch_srnn(self.config, self.test_set, action,
                                                                                  self.config.output_window_size)
            x_test[action] = encoder_inputs
            y_test[action] = decoder_outputs
            dec_in_test[action] = np.zeros([decoder_inputs.shape[0], 1, decoder_inputs.shape[2]])
            dec_in_test[action][:, 0, :] = decoder_inputs[:, 0, :]
        return [x_test, y_test, dec_in_test]

    def get_batch_srnn(self, config, data, action, target_seq_len):
        # Obtain SRNN test sequences using the specified random seeds

        frames = {}
        frames[action] = self.find_indices_srnn(data, action)

        batch_size = 8
        subject = 5
        source_seq_len = config.input_window_size

        seeds = [(action, (i % 2) + 1, frames[action][i]) for i in range(batch_size)]

        encoder_inputs = np.zeros((batch_size, source_seq_len - 1, config.input_size), dtype=float)
        decoder_inputs = np.zeros((batch_size, target_seq_len, config.input_size), dtype=float)
        decoder_outputs = np.zeros((batch_size, target_seq_len, config.input_size), dtype=float)

        for i in range(batch_size):
            _, subsequence, idx = seeds[i]
            idx = idx + 50

            data_sel = data[(subject, action, subsequence)]

            data_sel = data_sel[(idx - source_seq_len):(idx + target_seq_len), :]

            encoder_inputs[i, :, :] = data_sel[0:source_seq_len - 1, :]  # x_test
            decoder_inputs[i, :, :] = data_sel[source_seq_len - 1:(source_seq_len + target_seq_len - 1),
                                      :]  # decoder_in_test
            decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]  # y_test

        return [encoder_inputs, decoder_inputs, decoder_outputs]

    def find_indices_srnn(self, data, action):

        """
        Obtain the same action indices as in SRNN using a fixed random seed
        See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py
        """

        SEED = 1234567890
        rng = np.random.RandomState(SEED)

        subject = 5
        subaction1 = 1
        subaction2 = 2

        T1 = data[(subject, action, subaction1)].shape[0]
        T2 = data[(subject, action, subaction2)].shape[0]
        prefix, suffix = 50, 100

        idx = []
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))

        return idx


# The class below is discarded, just use human class to load h3.6m dataset instead
class H36mDataset(Dataset):
    """
    This dataset only contains lie algebra data
    Part of the code is copied from: https://github.com/BII-wushuang/Lie-Group-Motion-Prediction
    """

    def __init__(self, config, train=True):
        self.config = config
        self.train = train
        if train:
            subjects = [1, 6, 7, 8, 9, 11]
        else:
            subjects = [5]
        data_dir = './data/h3.6m/dataset'

        if config.filename == 'all':
            actions = ['directions', 'discussion', 'eating', 'greeting', 'phoning', 'posing', 'purchases', 'sitting',
                       'sittingdown', 'smoking', 'takingphoto', 'waiting', 'walking', 'walkingdog', 'walkingtogether']
        else:
            actions = [config.filename]

        set, complete_set = self.load_data(data_dir, subjects, actions)
        data_mean, data_std, dim_to_ignore, dim_to_use = self.normalization_stats(complete_set)

        if train:
            # Compute normalization stats
            data_mean, data_std, dim_to_ignore, dim_to_use = self.normalization_stats(complete_set)
            config.data_mean = data_mean
            config.data_std = data_std
            config.dim_to_ignore = dim_to_ignore
            config.dim_to_use = dim_to_use

            config.chain_idx = [np.array([0, 1, 2, 3, 4, 5]),
                                np.array([0, 6, 7, 8, 9, 10]),
                                np.array([0, 12, 13, 14, 15]),
                                np.array([13, 17, 18, 19, 22, 19, 21]),
                                np.array([13, 25, 26, 27, 30, 27, 29])]

        set = self.normalize_data(set, data_mean, data_std, dim_to_use)
        set_list = []
        for key in set.keys():
            set_list.append(set[key])

        self.data = set

    def load_data(self, data_dir, subjects, actions):
        """
           Copied from https://github.com/una-dinosauria/human-motion-prediction
        """
        train_data = {}
        complete_data = []
        for subj in subjects:
            for action in actions:
                for subact in [1, 2]:  # subactions
                    # print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(data_dir, subj, action, subact)
                    action_sequence = self.readCSVasFloat(filename)

                    n, d = action_sequence.shape
                    even_list = range(0, n, 2)

                    train_data[(subj, action, subact, 'even')] = action_sequence[even_list, :]

                if len(complete_data) == 0:
                    complete_data = copy.deepcopy(action_sequence)
                else:
                    complete_data = np.append(complete_data, action_sequence, axis=0)

        return [train_data, complete_data]

    def readCSVasFloat(self, filename):
        """
        Copied from https://github.com/una-dinosauria/human-motion-prediction
        """
        return_array = []
        lines = open(filename).readlines()
        for line in lines:
            line = line.strip().split(',')
            if len(line) > 0:
                return_array.append(np.array([np.float32(x) for x in line]))
        return_array = np.array(return_array)
        return return_array

    def normalization_stats(self, complete_data):
        """
        Copied from https://github.com/una-dinosauria/human-motion-prediction
        """
        data_mean = np.mean(complete_data, axis=0)
        data_std = np.std(complete_data, axis=0)

        dimensions_to_ignore = []
        dimensions_to_use = []

        dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
        dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))

        data_std[dimensions_to_ignore] = 1.0

        return [data_mean, data_std, dimensions_to_ignore, dimensions_to_use]

    def __getitem__(self, idx):

        sample = self.formatdata(self.data[idx])
        return sample

    def __len__(self):

        return len(self.data)


# The class below is discarded, testing on the mouse and fish dataset could be done using the same AnimalPredictionDataset dataset
class FishPredictionDataset(Dataset):

    def __init__(self, config):
        self.config = config
        self.lie_tsfm = LieTsfm(config)
        self.formatdata = FormatDataPre()
        if config.datatype == 'lie':
            x = []
            y = []
            for i in range(8):
                x_filename = './data/Fish/Test/x_test_lie/test_' + str(i) + '_lie.mat'
                y_filename = './data/Fish/Test/y_test_lie/test_' + str(i) + '_lie.mat'

                x_rawdata = sio.loadmat(x_filename)
                x_rawdata = x_rawdata[list(x_rawdata.keys())[3]]

                y_rawdata = sio.loadmat(y_filename)
                y_rawdata = y_rawdata[list(y_rawdata.keys())[3]]

                x.append(x_rawdata)
                y.append(y_rawdata)
            self.x = x
            self.y = y
        else:
            pass

    def __len__(self):

        return len(self.x)

    def __getitem__(self, idx):

        if self.config.datatype == 'lie':
            x_sample = self.lie_tsfm(self.x[idx])
            y_sample = self.lie_tsfm(self.y[idx])
        elif self.config.datatype == 'xyz':
            pass
        sample = self.formatdata(x_sample, y_sample)
        return sample

# The class below is discarded
class FishDataset(Dataset):

    def __init__(self, config, train=True):

        self.config = config
        self.train = train
        self.lie_tsfm = LieTsfm(config)
        self.formatdata = FormatData(config)
        if config.datatype == 'lie':
            train_path = './data/Fish/Train/train_lie/'
            tail = '_lie.mat'
        elif config.datatype == 'xyz':
            train_path = './data/Fish/Train/train_xyz/'
            tail = '_xyz.mat'
        if train:
            subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S7', 'S8']
        else:
            subjects = ['S6']

        set = []
        for id in subjects:
            filename = train_path + id + tail
            rawdata = sio.loadmat(filename)
            rawdata = rawdata[list(rawdata.keys())[3]]
            set.append(rawdata)
        self.data = set

    def __getitem__(self, idx):

        if self.config.datatype == 'lie':
            sample = self.data[idx][:, :-1, :3].reshape(self.data[idx].shape[0], -1)

            # sample = self.lie_tsfm(self.data[idx])
        elif self.config.datatype == 'xyz':
            pass
        sample = self.formatdata(sample, False)
        return sample

    def __len__(self):

        return len(self.data)

# The class below is discarded
class CSLPredictionDataset(object):

    def __init__(self, config):
        self.config = config
        self.x_test_dict = {}
        self.y_test_dict = {}
        self.dec_in_test_dict = {}
        if config.filename == 'all':
            self.actions = ['Circle', 'Square', 'Triangle']
        else:
            self.actions = [config.filename]

        if config.datatype == 'lie':

            for action in self.actions:
                x = []
                y = []
                for subact in [0, 1, 2, 3, 4]:
                    if config.datatype == 'lie':
                        x_filename = './data/CSL/Test/x_test_lie/test_' + str(action) +  '_' +str(subact) + '_lie.mat'
                        y_filename = './data/CSL/Test/y_test_lie/test_' + str(action) +  '_' +str(subact) + '_lie.mat'

                        x_rawdata = sio.loadmat(x_filename)
                        x_rawdata = x_rawdata[list(x_rawdata.keys())[3]]
                        x_rawdata = np.delete(x_rawdata[:, :, :3], [4, 11, 18], axis=1).reshape(x_rawdata.shape[0], -1)
                        x.append(x_rawdata)

                        y_rawdata = sio.loadmat(y_filename)
                        y_rawdata = y_rawdata[list(y_rawdata.keys())[3]]
                        y_rawdata = np.delete(y_rawdata[:, :, :3], [4, 11, 18], axis=1).reshape(y_rawdata.shape[0], -1)
                        y.append(y_rawdata)

                x = np.array(x)
                y = np.array(y)
                dec_in_test = np.reshape(x[:, -1, :], [x.shape[0], 1, x.shape[2]])
                x = x[:, 0:-1, :]
                self.x_test_dict[action] = x
                self.y_test_dict[action] = y
                self.dec_in_test_dict[action] = dec_in_test

        elif config.datatype == 'xyz':
            pass


    def get_data(self):

        return [self.x_test_dict, self.y_test_dict, self.dec_in_test_dict]

# The class below is discarded
class CSLDataset(Dataset):

    def __init__(self, config, train=True):
        self.config = config
        self.train = train
        self.lie_tsfm = LieTsfm(config)
        self.formatdata = FormatData(config)
        if config.datatype == 'lie':
            train_path = './data/CSL/Train/train_lie'
            tail = '_lie.mat'
        elif config.datatype == 'xyz':
            train_path = './data/CSL/Train/train_xyz'
            tail = '_xyz.mat'
        if train:
            subjects = ["S1", "S2", "S3", "S4", "S5"]
        else:
            subjects = ["S6"]

        if config.filename == 'all':
            actions = ['Circle', 'Square', 'Triangle']
        else:
            actions = [config.filename]

        set = []
        #complete_train = []
        # 这里应该还要遍历一个动作list的，测试就先不写
        for id in subjects:
            for action in actions:
                for i in range(5):
                    filename = '{0}/{1}_{2}_{3}_lie.mat'.format(train_path, id, action, i + 1)
                    rawdata = sio.loadmat(filename)
                    rawdata = rawdata[list(rawdata.keys())[3]]
                    rawdata = np.delete(rawdata[:, :, :3], [4, 11, 18], axis=1)
                    #rawdata = np.delete(rawdata[:, :, :3], [5, 11, 16, 21, 26], axis=1)
                    rawdata = rawdata.reshape(rawdata.shape[0], -1)
                    set.append(rawdata)

        #     if len(complete_train) == 0:
        #         complete_train = copy.deepcopy(set[-1])
        #     else:
        #         complete_train = np.append(complete_train, set[-1], axis=0)
        #
        # if not train and config.data_mean is None:
        #     print('Load train dataset first!')
        #
        # if train and config.datatype == 'lie':
        #     data_mean, data_std, dim_to_ignore, dim_to_use = utils.normalization_stats(complete_train)
        #     config.data_mean = data_mean
        #     config.data_std = data_std
        #     config.dim_to_ignore = dim_to_ignore
        #     config.dim_to_use = dim_to_use

        #set = utils.normalize_data(set, config.data_mean, config.data_std, list(range(0, 48)))

        self.data = set

    def __getitem__(self, idx):
        if self.config.datatype == 'lie':
            sample = self.data[idx]
        elif self.config.datatype == 'xyz':
            pass
        sample = self.formatdata(sample, False)
        return sample

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    import config

    config = config.TrainConfig(dataset='Human', datatype='smpl', action='walking', gpu=[0], training=True, visualize=False)
    data = CMUDataset(config, train=True)