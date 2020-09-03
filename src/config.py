# implemented by JunfengHu
# part of the code is brown from HRM
# create time 7/20/2019

import numpy as np


class TrainConfig(object):
    """Training Configurations"""
    train = True
    # run：9（12）
    # AMASS_labels = [
    # 00骨盆， 01左腿根，02右腿根， 03下背， 04左膝盖， 05右膝盖， 06上背， 07左踝，
    # 08右踝， 09胸膛， 10左脚趾， 11右脚趾， 12下颈，13左锁骨， 14右锁骨， 15上脖颈，
    # 16左臂根， 17右臂根， 18左肘， 19右肘， 20左腕，21右腕， 22左手，23右手]
    # 身体去除左右手，22节点
    # 实际上10， 11节点（左右脚趾）是全零，去除全零项，共20节点，的顺序应该是：
    # AMASS_labels = [
    # 00骨盆， 01左腿根，02右腿根， 03下背， 04左膝盖， 05右膝盖， 06上背， 07左踝，
    # 08右踝， 09胸膛， 10下颈，11左锁骨， 12右锁骨， 13上脖颈，
    # 14左臂根， 15右臂根， 16左肘， 17右肘， 18左腕，19右腕]
    # spine_id = [0, 3, 6, 9, 10, 13]    6 joints
    # left_arm_id = [11, 14, 16, 18]    4 joints
    # right_arm_id = [12, 15, 17, 19]    4 joints
    # left_leg_id = [1,4,7]    3 joints
    # right_leg_id = [2,5,8]    3 joints
    # data_root = '/mnt/DataDrive164/zhouhonghong/AMASS-CMU/CMU'

    # data_root = '/mnt/DataDrive164/zhouhonghong/AMASS-CMU/new_CMU'
    # data_root = '/mnt/DataDrive164/zhouhonghong/AMASS_selected/train/jump'
    # data_root = '/mnt/DataDrive164/zhouhonghong/AMASS_selected/train/kick'
    data_root = '/mnt/DataDrive164/zhouhonghong/AMASS_selected/train/walking_run'
    # data_root = '/mnt/DataDrive164/zhouhonghong/AMASS_selected/train/walk'
    # data_root = '/mnt/DataDrive164/zhouhonghong/AMASS_selected/train/locomotion'
    # subjects_train = ['08', '37', '45', '46', '47', '78']
    # subjects_train = ['dance_train']
    # subjects_train = ['salsa_dance_train_60']
    # subjects_train = ['bk']
    # walk
    # subjects_train = ['KIT_425', 'KIT_424', 'KIT_359', 'KIT_314', 'KIT_205', 'KIT_12', 'KIT_11', 'KIT_3', 'KIT_9', 'CMU_35']
    # subjects_train = ['CMU_74', 'KIT_3', 'KIT_513']
    # walking_run
    subjects_train = ['KIT_9', 'KIT_424', 'KIT_359', 'KIT_314', 'KIT_205', 'KIT_167']
    # subjects_train = ['05']
    # subjects_test = ['07']
    # subjects_test = ['dance_test']
    # subjects_test = ['salsa_dance_test_61']
    subjects_test = ['KIT_167']
    # walking_run
    subjects_test = ['KIT_3']
    # subjects_test = ['mm']
    # walk_and_run_train
    spine_id = [0, 3, 6, 9, 10, 13]
    left_arm_id = [11, 14, 16, 18]
    right_arm_id = [12, 15, 17, 19]
    left_leg_id = [1, 4, 7]
    right_leg_id = [2, 5, 8]
    changed = True   # 是否是zhouhonghong改过的方法
    input_window_size = 20  # Input window size during training
    output_window_size = 50  # Output window size during training
    hidden_size = 18  # Number of hidden units for HMR
    batch_size = 16  # Batch size for training
    learning_rate = 0.001  # Learning rate
    max_epoch = 200 #200  # Maximum training epochs 本来是500
    training_size = 100# 200  # Training iterations per epoch
    validation_size = 20  # Validation iterations per epoch
    restore = False  # Restore the trained weights or restart training from scratch
    longterm = False  # Whether we are doing super longterm prediction
    keep_prob = 0.6  # Keep probability for RNN cell weights
    context_window = 1  # Context window size in HMR, this para only applies to HMR
    encoder_recurrent_steps = 10  # Number of recurrent steps in HMR/ST_HRN
    decoder_recurrent_steps = 2  # Number of recurrent steps in ST-HMR decoder expect kinematics LSTM
    visualize = False               # visualize the predicted motion during testing

    models_name = ['HMR', 'ST_HRN']
    model = models_name[1]

    loss_name = ['l2', 'weightlie', 'HMRlie']
    loss = loss_name[1]
    """Only suitable for ST_HRN"""
    share_encoder_weights = True  # share encoder weight at each recurrent step, this param only applies to ST_HRN
    bone_dim = 3  # dimension of one bone representation, static in all datasets
    decoder_name = ['lstm', 'Kinematics_lstm']
    decoder = decoder_name[1]

    def __init__(self, dataset, datatype, action, gpu, training, visualize):
        self.device_ids = gpu  # index of GPU used to train the model
        self.train_model = training  # train or predict
        self.visualize = visualize  # visualize the predicted motion during testing
        self.dataset = dataset
        self.datatype = datatype
        self.filename = action
        # number of bones
        if dataset == 'Mouse':
            self.nbones = 4
            if self.decoder == 'Kinematics_lstm':
                self.decoder = self.decoder_name[0]
                print('You chose Kinematics_lstm as decoder, but lstm decoder is compatible for mouse dataset! Correct it automatically!!')
        elif dataset == 'Human':
            self.nbones = 20 #AMASS_CMU

        """Define kinematic chain configurations based on dataset class."""
        if self.dataset == 'Fish':
            self.filename = 'default'
            self.chain_config = [np.arange(0, 21)]
        elif self.dataset == 'Mouse':
            self.filename = 'default'
            self.chain_config = [np.arange(0, 5)]
        elif self.dataset == 'Human':
            # AMASS
            self.chain_config = [np.array([0, 1, 4, 7]),  # leg
                                 np.array([0, 2, 5, 8]),  # leg
                                 np.array([0, 3, 6, 9, 10, 13]),  # spine
                                 np.array([9, 11, 14, 16, 18]),  # arm
                                 np.array([9, 12, 15, 17, 19])]  # arm
            # H3.6M
            # self.chain_config = [np.array([0, 1, 2, 3, 4, 5]),  # leg
            #                      np.array([0, 6, 7, 8, 9, 10]),  # leg
            #                      np.array([0, 12, 13, 14, 15]),  # spine
            #                      np.array([13, 17, 18, 19, 22, 19, 21]),  # arm
            #                      np.array([13, 25, 26, 27, 30, 27, 29])]  # arm
            self.chain_loss_config = [np.array([1, 2, 3, 4, 5]),  # leg
                                 np.array([6, 7, 8, 9, 10]),  # leg
                                 np.array([0, 11, 12, 13, 14, 15]),  # spine
                                 np.array([16, 17, 18, 19, 20, 21, 22, 23]),  # arm
                                 np.array([24, 25, 26, 27, 28, 19, 30, 31])]  # arm
            self.training_chain_length = [9, 9, 18, 12, 12]
            # self.index = [[6, 7, 8, 9, 10, 11, 12, 13],
            #               [14, 15, 16, 17, 18, 19, 20, 21],
            #               [0, 1, 2, 3, 4, 5, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33],
            #               [34, 35, 36, 37, 38, 39, 40, 41, 42, 43],
            #               [44, 45, 46, 47, 48, 49, 50, 51, 52, 53]]
            self.index = [[3,4,5, 12,13,14, 21,22,23], #leg
                          [6,7,8, 15,16,17, 24,25,26], #leg
                          [0,1,2, 12,13,14, 18,19,20, 27,28,29, 30,31,32, 39,40,41], #spine
                          [33,34,35, 42,43,44, 48,49,50, 54,55,56], # arm
                          [36,37,38, 45,46,47, 51,52,53, 57,58,59]] #arm




