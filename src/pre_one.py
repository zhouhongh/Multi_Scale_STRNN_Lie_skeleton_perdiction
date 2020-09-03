import torch
import numpy as np
from choose_dataset import DatasetChooser
import utils
import scipy.io as sio
import config
import os
from ST_HRN import ST_HRN
# from ST_HRN_oringinal import ST_HRN
from argparse import ArgumentParser
import datetime
import joblib
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def normalize_data(data, data_mean, data_std, dim_to_use):
    """
     标准化 ，去除零值节点
    """
    data_out = []

    for idx in range(len(data)):
        data_out.append(np.divide((data[idx] - data_mean), data_std))
        data_out[-1] = data_out[-1][ dim_to_use]
    return data_out

def predict_one(config, input_pose, checkpoint_dir):
    # 更改预测时输出的视频帧长
    config.output_window_size = 90
    # 指定抽样后的起始帧 1: 470
    start_frame = 330
    datetime_p = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print('Start predict at time: ', datetime_p)
    output_dir = './predict_one_output/pose_'+datetime_p+'.npz'
    error_dir = './predict_one_output/error_'+datetime_p+'.mat'
    if not (os.path.exists('./predict_one_output/')):
        os.makedirs('./predict_one_output/')

    print("pose paras will be saved to: " + output_dir)
    print("errors will be saved to: " + error_dir)

    # This step is to get mean value,dim_to_use etc for norm and unnorm
    choose = DatasetChooser(config)
    _, _ = choose(train=True)

    x_test = {}
    y_test = {}
    dec_in_test = {}
    batch_size = 1
    source_seq_len = config.input_window_size
    target_seq_len = config.output_window_size
    encoder_inputs = np.zeros((batch_size, source_seq_len - 1, 60), dtype=float)
    decoder_inputs = np.zeros((batch_size, target_seq_len, 60), dtype=float)
    decoder_outputs = np.zeros((batch_size, target_seq_len, 60), dtype=float)

    data_seq = input_pose[start_frame:start_frame+source_seq_len+target_seq_len, :66] # 70（20+50）帧，每帧66个pose参数
    test_data = normalize_data(data_seq, config.data_mean, config.data_std, config.dim_to_use)

    for i in range(batch_size):
        encoder_inputs[i, :, :] = test_data[0:source_seq_len - 1][:]  # x_test
        decoder_inputs[i, :, :] = test_data[source_seq_len - 1:(source_seq_len + target_seq_len - 1)][:] # decoder_in_test
        decoder_outputs[i, :, :] = test_data[source_seq_len:][:]  # y_test, 相比于decoder_inputs 后移一位

    x_test['walking'] = encoder_inputs
    y_test['walking'] = decoder_outputs
    dec_in_test['walking'] = np.zeros([decoder_inputs.shape[0], 1, decoder_inputs.shape[2]])
    dec_in_test['walking'][:, 0, :] = decoder_inputs[:, 0, :]

    actions = list(x_test.keys()) # ['walking']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device {} will be used to save parameters'.format(device))
    net = ST_HRN(config)
    net.to(device)
    print('Total param number:' + str(sum(p.numel() for p in net.parameters())))
    print('Encoder param number:' + str(sum(p.numel() for p in net.encoder_cell.parameters())))
    print('Decoder param number:' + str(sum(p.numel() for p in net.decoder.parameters())))

    net = torch.nn.DataParallel(net)
    # dir = utils.get_file_list(checkpoint_dir)
    # print('Load model from:' + checkpoint_dir + dir[-1]) # 使用最后保存的checkpoint

    ## 用GPU时记得改成 map_location='cuda:0',用cpu时是 map_location=torch.device('cpu')
    # net.load_state_dict(torch.load(checkpoint_dir + dir[-1], map_location='cuda:0'))
    net.load_state_dict(torch.load(checkpoint_dir, map_location='cuda:0'))
    y_predict = {}
    with torch.no_grad():
        for act in actions:
            x_test_ = torch.from_numpy(x_test[act]).float().to(device)
            dec_in_test_ = torch.from_numpy(dec_in_test[act]).float().to(device)
            pred = net(x_test_, dec_in_test_, train=False)
            pred = pred.cpu().numpy()
            y_predict[act] = pred

    for act in actions:
        if config.datatype == 'smpl':
            mean_error, _ = utils.mean_euler_error(config, act, y_predict[act], y_test[act])
            sio.savemat(error_dir, dict([('error', mean_error)]))
            for i in range(y_predict[act].shape[0]):
                print(y_predict[act].shape[0])
                if config.dataset == 'Human':
                    y_p = utils.unNormalizeData(y_predict[act][i], config.data_mean, config.data_std, config.dim_to_ignore)
                    y_t = utils.unNormalizeData(y_test[act][i], config.data_mean, config.data_std, config.dim_to_ignore)

                # 保存
                np.savez(output_dir, y_p = y_p, y_t = y_t)



if __name__ == '__main__':
    # python pre_one.py --dataset Human --training False --action walking

    parser = ArgumentParser()
    parser.add_argument("--gpu", dest="gpu", default=[1], help="GPU device ids")
    parser.add_argument("--training", default=True, dest="training", help="train or test")
    parser.add_argument("--action", type=str, default='walking', dest="action", help="choose one action in the dataset:"
                                                                                 "h3.6m_actions = ['directions', 'discussion', 'eating', 'greeting', 'phoning', 'posing', 'purchases', 'sitting',"
                                                                                 "'sittingdown', 'smoking', 'takingphoto', 'waiting', 'walking', 'walkingdog', 'walkingtogether']"
                                                                                 "'all means all of the above")
    parser.add_argument("--dataset", type=str, required=True, dest="dataset",
                        help="choose dataset from 'Human' or 'Mouse'")
    parser.add_argument("--datatype", type=str, default='smpl', dest="datatype", help="only lie is usable")
    parser.add_argument("--visualize", type=bool, default=False, dest="visualize",
                        help="visualize the prediction or not ")
    args = parser.parse_args()
    config = config.TrainConfig(args.dataset, args.datatype, args.action, args.gpu, args.training, args.visualize)
    # checkpoint_dir, _ = utils.create_directory(config)

    # checkpoint_dir = '/home/ubuntu/users/zhouhonghong/codes/articulated-objects-motion-prediction_CMU/src/zhh_checkpoint_CMU/Human/smpl_weightlieloss_ST_HRN_RecurrentSteps=10_hiddenSize=18_decoder_name=Kinematics_lstm/walking_run_orin_filt/inputWindow=20_outputWindow=50/Epoch_13.pth'
    # checkpoint_dir = '/home/ubuntu/users/zhouhonghong/codes/articulated-objects-motion-prediction_CMU/src/zhh_checkpoint_CMU/Human/smpl_l2loss_ST_HRN_RecurrentSteps=10_hiddenSize=18_decoder_name=Kinematics_lstm/run/inputWindow=20_outputWindow=50/Epoch_97.pth'
    # checkpoint_dir = '/home/ubuntu/users/zhouhonghong/codes/articulated-objects-motion-prediction_CMU/src/zhh_checkpoint_CMU/Human/smpl_weightlieloss_ST_HRN_RecurrentSteps=10_hiddenSize=18_decoder_name=Kinematics_lstm/walking_run_cha_filt2/inputWindow=20_outputWindow=50/Epoch_30.pth'
    #dance_2_cha
    # checkpoint_dir = '/home/ubuntu/users/zhouhonghong/codes/articulated-objects-motion-prediction_CMU/src/zhh_checkpoint_CMU/Human/smpl_weightlieloss_ST_HRN_RecurrentSteps=10_hiddenSize=18_decoder_name=Kinematics_lstm/salsa_dance_2_cha/inputWindow=20_outputWindow=50/Epoch_88.pth'
    # ballet_dance
    # checkpoint_dir = '/home/ubuntu/users/zhouhonghong/codes/articulated-objects-motion-prediction_CMU/src/zhh_checkpoint_CMU/Human/smpl_l2loss_ST_HRN_RecurrentSteps=10_hiddenSize=18_decoder_name=Kinematics_lstm/new_dance/inputWindow=20_outputWindow=50/Epoch_62.pth'
    # walking_run
    checkpoint_dir = '/home/ubuntu/users/zhouhonghong/codes/articulated-objects-motion-prediction_CMU/src/zhh_checkpoint_CMU/Human/smpl_weightlieloss_ST_HRN_RecurrentSteps=10_hiddenSize=18_decoder_name=Kinematics_lstm/walking_run_cha_filt2/inputWindow=20_outputWindow=50/Epoch_30.pth'
    #walking
    # checkpoint_dir = '/home/ubuntu/users/zhouhonghong/codes/articulated-objects-motion-prediction_CMU/src/zhh_checkpoint_CMU/Human/smpl_weightlieloss_ST_HRN_RecurrentSteps=10_hiddenSize=18_decoder_name=Kinematics_lstm/walking/inputWindow=20_outputWindow=50/Epoch_48.pth'
    #若使用系统默认checkpoint还需要将predict_one函数中的注释代码解除注释
    # 输入姿势[frames,72]

    # zhh_para = joblib.load("zhou_walking.pkl")
    # input_pose = zhh_para[1]['pose']
    # input_data_dir = '/mnt/DataDrive164/zhouhonghong/zhh2/1/vibe_output.pkl'

    input_data_dir = '/mnt/DataDrive164/zhouhonghong/zhou_20200901/vibe_output.pkl'
    input_data = joblib.load(input_data_dir)
    input_pose = input_data[1]['pose']
    print('input_pose.shape', input_pose.shape)

    # input_data_dir = '/mnt/DataDrive164/zhouhonghong/AMASS-CMU/new_CMU/dance_test/05_04_poses.npz'
    # input_data_dir = '/mnt/DataDrive164/zhouhonghong/AMASS-CMU/new_CMU/salsa_dance_test_61/61_01_poses.npz'
    # input_pose = np.load(input_data_dir)['poses']
    predict_one(config, input_pose, checkpoint_dir)