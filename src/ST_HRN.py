# implemented by JunfengHu
# create time: 7/20/2019

import torch
import torch.nn as nn
import torch.nn.functional as F

## zhh
class ST_HRN(nn.Module):



    def __init__(self, config):
        super().__init__()

        self.config = config
        self.encoder_cell = torch.nn.ModuleList()
        # init encoder
        if config.share_encoder_weights is False:
            for i in range(config.encoder_recurrent_steps):  ##叠加十个RNN
                self.encoder_cell.append(EncoderCell(config))
        else:
            shared_encoder = EncoderCell(config)  ##十个RNN之间共享参数
            for i in range(config.encoder_recurrent_steps):
                self.encoder_cell.append(shared_encoder)

        # init decoder
        if config.decoder == 'st_lstm':
            print('Use ST_LSTM as decoder.')
            self.decoder = ST_LSTM(config)
        elif config.decoder == 'lstm':
            print('Use LSTM as decoder.')
            self.decoder = LSTM_decoder(config)
        elif config.decoder == 'Kinematics_lstm':
            print('Use Kinematics_LSTM as decoder.')
            self.decoder = Kinematics_LSTM_decoder(config)

        self.weights_in = torch.nn.Parameter(torch.empty(config.input_size,
                                      int(config.input_size/config.bone_dim*config.hidden_size)).uniform_(-0.04, 0.04)) ##[54,324]
        self.bias_in = torch.nn.Parameter(torch.empty(int(config.input_size/config.bone_dim*config.hidden_size)).uniform_(-0.04, 0.04)) ##均匀分布中采样填充

    def forward(self, encoder_inputs, decoder_inputs, train):
        """
        The decoder and encoder wrapper.
        :param encoder_inputs:
        :param decoder_inputs:
        :param train: train or test the model
        :return:  predictions of human motion
        """

        # [batch, config.input_window_size-1, input_size/bone_dim*hidden_size]
        h = torch.matmul(encoder_inputs, self.weights_in) + self.bias_in  ##([6,49,54],[54,324])+[324]=[6,49,324]
        # [batch, config.input_window_size-1, hidden_size]     [6,49,18,18]
        h = h.view([h.shape[0], h.shape[1], int(h.shape[2]/self.config.hidden_size), self.config.hidden_size])
        # [batch, frames,  nbones, hidden_state]
        #h = F.dropout(h, self.config.keep_prob, train)
        c_h = torch.empty_like(h)
        c_h.copy_(h)
        #c_h = F.dropout(c_h, self.config.keep_prob, train)

        #p = torch.empty_like(h)
        #p.copy_(h)
        # [6,49,18,3],22段骨骼，每段骨骼三维
        p = encoder_inputs.view([encoder_inputs.shape[0], encoder_inputs.shape[1], int(encoder_inputs.shape[2]/self.config.bone_dim), self.config.bone_dim])
        # init global states
        # [batch, nbones, hidden_size], 在时间维求了平均
        g_t = torch.mean(h, 2, keepdim=True).expand_as(h)
        c_g_t = torch.mean(c_h, 2, keepdim=True).expand_as(c_h)
        # test_h = h[:,:(self.config.input_window_size - 1)//3,:, :]
        g_t1 = torch.mean(h[:,:(self.config.input_window_size - 1)//3, :, :], 1, keepdim=True).expand_as(h)
        g_t2 = torch.mean(h[:, (self.config.input_window_size - 1)//3:2*(self.config.input_window_size - 1)//3, :, :], 1, keepdim=True).expand_as(
            h)
        g_t3 = torch.mean(h[:, 2*(self.config.input_window_size - 1)//3:, :, :], 1, keepdim=True).expand_as(
            h)
        c_g_t1 = torch.mean(c_h[:, :(self.config.input_window_size - 1) // 3, :, :], 1, keepdim=True).expand_as(
            c_h)
        c_g_t2 = torch.mean(
            c_h[:, (self.config.input_window_size - 1) // 3:2 * (self.config.input_window_size - 1) // 3, :, :], 1,
            keepdim=True).expand_as(
            c_h)
        c_g_t3 = torch.mean(c_h[:, 2 * (self.config.input_window_size - 1) // 3:, :, :], 1, keepdim=True).expand_as(
            c_h)



        # [batch, input_window_size-1, hidden_size]，在空间维（骨骼）求了平均
        g_s = torch.mean(h, 1, keepdim=True).expand_as(h)
        c_g_s = torch.mean(c_h, 1, keepdim=True).expand_as(c_h)

        g_s_spine = torch.mean(h[:, :, self.config.spine_id, :], 2, keepdim = True).expand_as(h)
        g_s_left_arm = torch.mean(h[:, :, self.config.left_arm_id, :], 2, keepdim = True).expand_as(h)
        g_s_right_arm = torch.mean(h[:, :, self.config.right_arm_id, :], 2, keepdim=True).expand_as(h)
        g_s_left_leg = torch.mean(h[:, :, self.config.left_leg_id, :], 2, keepdim=True).expand_as(h)
        g_s_right_leg = torch.mean(h[:, :, self.config.right_leg_id, :], 2, keepdim=True).expand_as(h)

        c_g_s_spine = torch.mean(c_h[:, :, self.config.spine_id, :], 2, keepdim = True).expand_as(c_h)
        c_g_s_left_arm = torch.mean(c_h[:, :, self.config.left_arm_id, :], 2, keepdim = True).expand_as(c_h)
        c_g_s_right_arm = torch.mean(c_h[:, :, self.config.right_arm_id, :], 2, keepdim=True).expand_as(c_h)
        c_g_s_left_leg = torch.mean(c_h[:, :, self.config.left_leg_id, :], 2, keepdim=True).expand_as(c_h)
        c_g_s_right_leg = torch.mean(c_h[:, :, self.config.right_leg_id, :], 2, keepdim=True).expand_as(c_h)



        #     def forward(self, h, c_h, p, g_t,c_g_t,g_t1, g_t2, g_t3,c_g_t1, c_g_t2,c_g_t3, g_s, c_g_s,g_s_spine, c_g_s_spine
        #                 ,g_s_left_arm,c_g_s_left_arm, g_s_right_arm,c_g_s_right_arm,g_s_left_leg,c_g_s_left_leg
        #                 ,g_s_right_leg,c_g_s_right_leg,train):
        #  return hidden_states, cell_states, global_t_state, g_t, c_g_t, g_t1, g_t2, g_t3,c_g_t1, c_g_t2,c_g_t3,
        #  g_s, c_g_s,g_s_spine, c_g_s_spine,g_s_left_arm,c_g_s_left_arm, g_s_right_arm,c_g_s_right_arm,
        #  g_s_left_leg,c_g_s_left_leg,g_s_right_leg,c_g_s_right_leg
        for rec in range(self.config.encoder_recurrent_steps):
            hidden_states, cell_states, global_t_state, g_t, c_g_t, g_t1,g_t2, g_t3,c_g_t1, c_g_t2,c_g_t3, g_s, c_g_s, \
            g_s_spine, c_g_s_spine, g_s_left_arm, c_g_s_left_arm, g_s_right_arm, c_g_s_right_arm, \
            g_s_left_leg, c_g_s_left_leg, g_s_right_leg, c_g_s_right_leg \
                = self.encoder_cell[rec](h, c_h, p, g_t, c_g_t, g_t1, g_t2, g_t3,c_g_t1, c_g_t2,c_g_t3,
                                         g_s, c_g_s,g_s_spine, c_g_s_spine, g_s_left_arm,c_g_s_left_arm, g_s_right_arm,c_g_s_right_arm, \
                                         g_s_left_leg, c_g_s_left_leg,g_s_right_leg,c_g_s_right_leg,train)
        prediction = self.decoder(hidden_states, cell_states, global_t_state, decoder_inputs)
        # return prediction, hidden_states,g_s_spine,g_s_left_arm,g_s_right_arm,g_s_left_leg,g_s_right_leg
        return prediction



class EncoderCell(nn.Module):
    """
    ST_HRN encoder
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        """h update gates"""
        # input  gate
        #self.Ui = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Ui = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.bone_dim, self.config.hidden_size))
        self.Wti = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wsi = torch.nn.Parameter(torch.randn((self.config.input_window_size - 1), self.config.hidden_size * 3, self.config.hidden_size))
        self.Zti = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt1i = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt2i = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt3i = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsi = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zssi = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslai = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrai = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslli = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrli = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bi = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # left time forget gate
        #self.Ult = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Ult = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.bone_dim, self.config.hidden_size))
        self.Wtlt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wslt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Ztlt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt1lt = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt2lt = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt3lt = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsslt = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslalt = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsralt = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslllt = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrllt = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.blt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # forward time forget gate
        #self.Uft = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Uft = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.bone_dim, self.config.hidden_size))
        self.Wtft = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wsft = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Ztft = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt1ft = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt2ft = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt3ft = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsft = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zssft = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslaft = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsraft = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsllft = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrlft = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bft = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # right time forget gate
        self.Urt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.bone_dim, self.config.hidden_size))
        #self.Urt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Wtrt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wsrt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Ztrt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt1rt = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt2rt = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt3rt = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zssrt = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslart = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrart = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsllrt = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrlrt = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.brt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # space forget gate
        #self.Us = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Us = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.bone_dim, self.config.hidden_size))
        self.Wts = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wss = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Zts = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt1s = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt2s = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt3s = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zss = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsss = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslas = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsras= torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslls = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrls = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bs = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # left space forget gate
        # self.Us = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Uls = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.bone_dim, self.config.hidden_size))
        self.Wtls = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wsls = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Ztls = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt1ls = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt2ls = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt3ls = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsls = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zssls = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslals = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrals = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsllls = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrlls = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bls = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # right space forget gate
        # self.Us = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Urs = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.bone_dim, self.config.hidden_size))
        self.Wtrs = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wsrs = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Ztrs = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt1rs = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt2rs = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt3rs = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrs = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zssrs = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslars = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrars = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsllrs = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrlrs = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.brs = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # global time forgate gate
        #self.Ugt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Ugt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.bone_dim, self.config.hidden_size))
        self.Wtgt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wsgt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Ztgt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt1gt = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt2gt = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt3gt = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsgt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zssgt = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslagt = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsragt = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsllgt = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrlgt = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgt = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))


        # global time forgate gate for scale 1
        self.Ugt1 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.bone_dim, self.config.hidden_size))
        self.Wtgt1 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wsgt1 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Ztgt1 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt1gt1 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt2gt1 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt3gt1 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsgt1 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zssgt1 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslagt1 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsragt1 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsllgt1 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrlgt1 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgt1 = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # global time forgate gate for scale 2
        self.Ugt2 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.bone_dim, self.config.hidden_size))
        self.Wtgt2 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wsgt2 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Ztgt2 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt1gt2 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt2gt2 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt3gt2 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsgt2 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zssgt2 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslagt2 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsragt2 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsllgt2 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrlgt2 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgt2 = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # global time forgate gate for scale 3
        self.Ugt3 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.bone_dim, self.config.hidden_size))
        self.Wtgt3 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wsgt3 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Ztgt3 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt1gt3 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt2gt3 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt3gt3 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsgt3 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zssgt3 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslagt3 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsragt3 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsllgt3 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrlgt3 = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgt3 = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))


        # global space forgate gate
        #self.Ugs = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Ugs = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.bone_dim, self.config.hidden_size))
        self.Wtgs = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wsgs = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Ztgs = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt1gs = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt2gs = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt3gs = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsgs = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zssgs = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslags = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrags = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsllgs = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrlgs = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgs = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # global space forgate gate for spine
        #self.Ugs = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Ugss = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.bone_dim, self.config.hidden_size))
        self.Wtgss = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wsgss = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Ztgss = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt1gss = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt2gss = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt3gss = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsgss = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zssgss = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslagss = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsragss = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsllgss = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrlgss = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgss = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # global space forgate gate for left arm
        #self.Ugs = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Ugsla = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.bone_dim, self.config.hidden_size))
        self.Wtgsla = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wsgsla = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Ztgsla = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt1gsla = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt2gsla = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt3gsla = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsgsla = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zssgsla = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslagsla = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsragsla = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsllgsla = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrlgsla = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgsla = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # global space forgate gate for right arm
        #self.Ugs = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Ugsra = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.bone_dim, self.config.hidden_size))
        self.Wtgsra = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wsgsra = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Ztgsra = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt1gsra= torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt2gsra = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt3gsra = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsgsra = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zssgsra = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslagsra = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsragsra = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsllgsra = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrlgsra = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgsra = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # global space forgate gate for left leg
        #self.Ugs = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Ugsll = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.bone_dim, self.config.hidden_size))
        self.Wtgsll = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wsgsll = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Ztgsll = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt1gsll = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt2gsll = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt3gsll = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsgsll = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zssgsll = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslagsll = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsragsll = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsllgsll = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrlgsll = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgsll = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # global space forgate gate for right leg
        #self.Ugs = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Ugsrl = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.bone_dim, self.config.hidden_size))
        self.Wtgsrl = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wsgsrl = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Ztgsrl = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt1gsrl = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt2gsrl = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt3gsrl = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsgsrl = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zssgsrl = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslagsrl = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsragsrl = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsllgsrl = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrlgsrl = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgsrl = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))


        # output gate
        #self.Uo = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Uo = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.bone_dim, self.config.hidden_size))
        self.Wto = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wso = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Zto = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt1o = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt2o = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt3o = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zso = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsso = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslao = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrao = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsllo = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrlo = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bo = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # c_hat gate
        #self.Uc = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Uc = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.bone_dim, self.config.hidden_size))
        self.Wtc = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Wsc = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size * 3, self.config.hidden_size))
        self.Ztc = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt1c = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt2c = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zt3c = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsc = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zssc = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zslac = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrac = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsllc = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zsrlc = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bc = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        """g_t update gates"""
        # forget gates for h
        self.Wgtf = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgtf = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgtf = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # forget gate for g
        self.Wgtg = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgtg = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgtg = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # output gate
        self.Wgto = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgto = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgto = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        """g_t1 update gates"""
        # forget gates for h
        self.Wgt1f = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgt1f = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgt1f = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # forget gate for g_t1
        self.Wgt1g = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgt1g = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgt1g = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # output gate
        self.Wgt1o = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgt1o = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgt1o = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        """g_t2 update gates"""
        # forget gates for h
        self.Wgt2f = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgt2f = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgt2f = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # forget gate for g_t2
        self.Wgt2g = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgt2g = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgt2g = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # output gate
        self.Wgt2o = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgt2o = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgt2o = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        """g_t3 update gates"""
        # forget gates for h
        self.Wgt3f = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgt3f = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgt3f = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # forget gate for g_t3
        self.Wgt3g = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgt3g = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgt3g = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # output gate
        self.Wgt3o = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgt3o = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgt3o = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        """g_s update gates"""
        # forget gates for h
        self.Wgsf = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgsf = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgsf = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # forget gate for g
        self.Wgsg = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgsg = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgsg = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # output gate
        self.Wgso = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgso = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgso = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        """g_s_spine update gates"""
        # forget gates for h
        self.Wgssf = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgssf = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgssf = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # forget gate for g_s_spine
        self.Wgssg = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgssg = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgssg = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # output gate
        self.Wgsso = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgsso = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgsso = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        """g_s_left_arm update gates"""
        # forget gates for h
        self.Wgslaf = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgslaf = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgslaf = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # forget gate for g_s_left_arm
        self.Wgslag = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgslag = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgslag = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # output gate
        self.Wgslao = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgslao = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgslao = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        """g_s_right_arm update gates"""
        # forget gates for h
        self.Wgsraf = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgsraf = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgsraf = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # forget gate for g_s_right_arm
        self.Wgsrag = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgsrag = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgsrag = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # output gate
        self.Wgsrao = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgsrao = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgsrao = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        """g_s_left_leg update gates"""
        # forget gates for h
        self.Wgsllf = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgsllf = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgsllf = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # forget gate for g_s_left_leg
        self.Wgsllg = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgsllg = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgsllg = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # output gate
        self.Wgsllo = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgsllo = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgsllo = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        """g_s_right_leg update gates"""
        # forget gates for h
        self.Wgsrlf = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgsrlf = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgsrlf = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # forget gate for g_s_right_leg
        self.Wgsrlg = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgsrlg = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgsrlg = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

        # output gate
        self.Wgsrlo = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.Zgsrlo = torch.nn.Parameter(
            torch.randn(self.config.input_window_size - 1, self.config.hidden_size, self.config.hidden_size))
        self.bgsrlo = torch.nn.Parameter(torch.randn(self.config.input_window_size - 1, 1, self.config.hidden_size))

    def forward(self, h, c_h, p, g_t,c_g_t,g_t1, g_t2, g_t3,c_g_t1, c_g_t2,c_g_t3, g_s, c_g_s,g_s_spine, c_g_s_spine
                ,g_s_left_arm,c_g_s_left_arm, g_s_right_arm,c_g_s_right_arm,g_s_left_leg,c_g_s_left_leg
                ,g_s_right_leg,c_g_s_right_leg,train):
        """
        :param h: hidden states of [batch, input_window_size-1, nbones, hidden_size]
        :param c_h: cell states of  [batch, input_window_size-1, nbones, hidden_size]
        :param p: pose of  [batch, input_window_size-1, nbones, hidden_size]
        :param g_t: [batch, input_window_size-1, nbones, hidden_size]
        :param c_g_t: [batch, input_window_size-1, nbones, hidden_size]
        :param g_s: [batch, input_window_size-1, nbones, hidden_size]
        :param c_g_s: [batch, input_window_size-1, nbones, hidden_size]
        :param train: control dropout
        :return: hidden_states, cell_states, global_t_state, g_t, c_g_t, g_s, c_g_s
        hidden_states, cell_states, global_t_state at last encoding recurrent will be used in decoder.
        g_t, c_g_t, g_s, c_g_s will be used for next recurrent
        """

        padding_t = torch.zeros_like(h[:, 0:1, :, :])
        padding_s = torch.zeros_like(h[:, :, 0:1, :])

        """Update h"""

        h_t_before = torch.cat((padding_t, h[:, :-1, :, :]), dim=1)
        h_t_after = torch.cat((h[:, 1:, :, :], padding_t), dim=1)
        # [batch, input_window_size-1, nbones, hidden_size*3]
        h_t_before_after = torch.cat((h_t_before, h, h_t_after), dim=3) ##将隐藏向量连起来

        c_t_before = torch.cat((padding_t, c_h[:, :-1, :, :]), dim=1)
        c_t_after = torch.cat((c_h[:, 1:, :, :], padding_t), dim=1)
        # [batch, input_window_size-1, nbones, hidden_size*3]

        h_s_before = torch.cat((padding_s, h[:, :, :-1, :]), dim=2)
        c_s_before = torch.cat((padding_s, c_h[:, :, :-1, :]), dim=2)

        h_s_after = torch.cat((h[:, :, 1:, :], padding_s), dim=2)
        c_s_after = torch.cat((c_h[:, :, 1:, :], padding_s), dim=2)
        h_s_before_after = torch.cat((h_s_before, h, h_s_after), dim=3)

        # forget gates for h
        i_n = torch.sigmoid(torch.matmul(p, self.Ui) + torch.matmul(h_t_before_after, self.Wti)
                            + torch.matmul(h_s_before_after, self.Wsi) + torch.matmul(g_t, self.Zti)
                            + torch.matmul(g_s, self.Zsi) + torch.matmul(g_t1, self.Zt1i)
                            + torch.matmul(g_t2, self.Zt2i) + torch.matmul(g_t3, self.Zt3i)
                            + torch.matmul(g_s_spine, self.Zssi) + torch.matmul(g_s_left_arm, self.Zslai)
                            + torch.matmul(g_s_right_arm, self.Zsrai) + torch.matmul(g_s_left_leg, self.Zslli)
                            + torch.matmul(g_s_right_leg, self.Zsrli) + self.bi)
        #bi:49,1,18;  # g_t1:12,49,18,18   g_s:12,49,18,18, g_t:12,49,18,18 ,g_s_spine:12,49,4,18
        #Zsi:49,18,18,3; Zssi:49,18,18,; Zt1i:16,18,18
        f_lt_n = torch.sigmoid(torch.matmul(p, self.Ult) + torch.matmul(h_t_before_after, self.Wtlt)
                               + torch.matmul(h_s_before_after, self.Wslt) + torch.matmul(g_t, self.Ztlt)
                               + torch.matmul(g_s, self.Zslt) + torch.matmul(g_t1, self.Zt1lt)
                               + torch.matmul(g_t2, self.Zt2lt) + torch.matmul(g_t3, self.Zt3lt)
                               + torch.matmul(g_s_spine, self.Zsslt) + torch.matmul(g_s_left_arm, self.Zslalt)
                               + torch.matmul(g_s_right_arm, self.Zsralt) + torch.matmul(g_s_left_leg, self.Zslllt)
                               + torch.matmul(g_s_right_leg, self.Zsrllt) + self.blt)
        f_ft_n = torch.sigmoid(torch.matmul(p, self.Uft) + torch.matmul(h_t_before_after, self.Wtft)
                               + torch.matmul(h_s_before_after, self.Wsft) + torch.matmul(g_t, self.Ztft)
                               + torch.matmul(g_s, self.Zsft) + torch.matmul(g_t1, self.Zt1ft)
                               + torch.matmul(g_t2, self.Zt2ft) + torch.matmul(g_t3, self.Zt3ft)
                               + torch.matmul(g_s_spine, self.Zssft) + torch.matmul(g_s_left_arm, self.Zslaft)
                               + torch.matmul(g_s_right_arm, self.Zsraft) + torch.matmul(g_s_left_leg, self.Zsllft)
                               + torch.matmul(g_s_right_leg, self.Zsrlft) + self.bft)
        f_rt_n = torch.sigmoid(torch.matmul(p, self.Urt) + torch.matmul(h_t_before_after, self.Wtrt)
                               + torch.matmul(h_s_before_after, self.Wsrt) + torch.matmul(g_t, self.Ztrt)
                               + torch.matmul(g_s, self.Zsrt) + torch.matmul(g_t1, self.Zt1rt)
                               + torch.matmul(g_t2, self.Zt2rt) + torch.matmul(g_t3, self.Zt3rt)
                               + torch.matmul(g_s_spine, self.Zssrt) + torch.matmul(g_s_left_arm, self.Zslart)
                               + torch.matmul(g_s_right_arm, self.Zsrart) + torch.matmul(g_s_left_leg, self.Zsllrt)
                               + torch.matmul(g_s_right_leg, self.Zsrlrt) + self.brt)
        f_s_n = torch.sigmoid(torch.matmul(p, self.Us) + torch.matmul(h_t_before_after, self.Wts)
                              + torch.matmul(h_s_before_after, self.Wss) + torch.matmul(g_t, self.Zts)
                              + torch.matmul(g_s, self.Zss) + torch.matmul(g_t1, self.Zt1s)
                              + torch.matmul(g_t2, self.Zt2s) + torch.matmul(g_t3, self.Zt3s)
                              + torch.matmul(g_s_spine, self.Zsss) + torch.matmul(g_s_left_arm, self.Zslas)
                              + torch.matmul(g_s_right_arm, self.Zsras) + torch.matmul(g_s_left_leg, self.Zslls)
                              + torch.matmul(g_s_right_leg, self.Zsrls) + self.bs)

        f_ls_n = torch.sigmoid(torch.matmul(p, self.Uls) + torch.matmul(h_t_before_after, self.Wtls)
                               + torch.matmul(h_s_before_after, self.Wsls) + torch.matmul(g_t, self.Ztls)
                               + torch.matmul(g_s, self.Zsls) + torch.matmul(g_t1, self.Zt1ls)
                               + torch.matmul(g_t2, self.Zt2ls) + torch.matmul(g_t3, self.Zt3ls)
                               + torch.matmul(g_s_spine, self.Zssls) + torch.matmul(g_s_left_arm, self.Zslals)
                               + torch.matmul(g_s_right_arm, self.Zsrals) + torch.matmul(g_s_left_leg, self.Zsllls)
                               + torch.matmul(g_s_right_leg, self.Zsrlls) + self.bls)

        f_rs_n = torch.sigmoid(torch.matmul(p, self.Urs) + torch.matmul(h_t_before_after, self.Wtrs)
                              + torch.matmul(h_s_before_after, self.Wsrs) + torch.matmul(g_t, self.Ztrs)
                              + torch.matmul(g_s, self.Zsrs) + torch.matmul(g_t1, self.Zt1rs)
                              + torch.matmul(g_t2, self.Zt2rs) + torch.matmul(g_t3, self.Zt3rs)
                              + torch.matmul(g_s_spine, self.Zssrs) + torch.matmul(g_s_left_arm, self.Zslars)
                              + torch.matmul(g_s_right_arm, self.Zsrars) + torch.matmul(g_s_left_leg, self.Zsllrs)
                              + torch.matmul(g_s_right_leg, self.Zsrlrs) + self.brs)




        f_gt_n = torch.sigmoid(torch.matmul(p, self.Ugt) + torch.matmul(h_t_before_after, self.Wtgt)
                               + torch.matmul(h_s_before_after, self.Wsgt) + torch.matmul(g_t, self.Ztgt)
                               + torch.matmul(g_s, self.Zsgt) + torch.matmul(g_t1, self.Zt1gt)
                               + torch.matmul(g_t2, self.Zt2gt) + torch.matmul(g_t3, self.Zt3gt)
                               + torch.matmul(g_s_spine, self.Zssgt) + torch.matmul(g_s_left_arm, self.Zslagt)
                               + torch.matmul(g_s_right_arm, self.Zsragt) + torch.matmul(g_s_left_leg, self.Zsllgt)
                               + torch.matmul(g_s_right_leg, self.Zsrlgt) + self.bgt)
        f_gs_n = torch.sigmoid(torch.matmul(p, self.Ugs) + torch.matmul(h_t_before_after, self.Wtgs)
                               + torch.matmul(h_s_before_after, self.Wsgs) + torch.matmul(g_t, self.Ztgs)
                               + torch.matmul(g_s, self.Zsgs) + torch.matmul(g_t1, self.Zt1gs)
                               + torch.matmul(g_t2, self.Zt2gs) + torch.matmul(g_t3, self.Zt3gs)
                               + torch.matmul(g_s_spine, self.Zssgs) + torch.matmul(g_s_left_arm, self.Zslags)
                               + torch.matmul(g_s_right_arm, self.Zsrags) + torch.matmul(g_s_left_leg, self.Zsllgs)
                               + torch.matmul(g_s_right_leg, self.Zsrlgs) + self.bgs)

        # 添加g_t1,g_t2,g_t3,g_s_spine,g_s_left_arm,g_s_right_arm,g_s_left_leg, g_s_right_leg的遗忘门：
        # f_gt1_n, f_gt2_n, f_gt3_n, f_gss_n, f_gsla_n, f_gsra_n, f_gsll_n, f_gsrl_n

        f_gt1_n = torch.sigmoid(torch.matmul(p, self.Ugt1) + torch.matmul(h_t_before_after, self.Wtgt1)
                               + torch.matmul(h_s_before_after, self.Wsgt1) + torch.matmul(g_t, self.Ztgt1)
                               + torch.matmul(g_s, self.Zsgt1) + torch.matmul(g_t1, self.Zt1gt1)
                               + torch.matmul(g_t2, self.Zt2gt1) + torch.matmul(g_t3, self.Zt3gt1)
                               + torch.matmul(g_s_spine, self.Zssgt1) + torch.matmul(g_s_left_arm, self.Zslagt1)
                               + torch.matmul(g_s_right_arm, self.Zsragt1) + torch.matmul(g_s_left_leg, self.Zsllgt1)
                               + torch.matmul(g_s_right_leg, self.Zsrlgt1) + self.bgt1)
        f_gt2_n = torch.sigmoid(torch.matmul(p, self.Ugt2) + torch.matmul(h_t_before_after, self.Wtgt2)
                                + torch.matmul(h_s_before_after, self.Wsgt2) + torch.matmul(g_t, self.Ztgt2)
                                + torch.matmul(g_s, self.Zsgt2) + torch.matmul(g_t1, self.Zt1gt2)
                                + torch.matmul(g_t2, self.Zt2gt2) + torch.matmul(g_t3, self.Zt3gt2)
                                + torch.matmul(g_s_spine, self.Zssgt2) + torch.matmul(g_s_left_arm, self.Zslagt2)
                                + torch.matmul(g_s_right_arm, self.Zsragt2) + torch.matmul(g_s_left_leg, self.Zsllgt2)
                                + torch.matmul(g_s_right_leg, self.Zsrlgt2) + self.bgt2)
        f_gt3_n = torch.sigmoid(torch.matmul(p, self.Ugt3) + torch.matmul(h_t_before_after, self.Wtgt3)
                                + torch.matmul(h_s_before_after, self.Wsgt3) + torch.matmul(g_t, self.Ztgt3)
                                + torch.matmul(g_s, self.Zsgt3) + torch.matmul(g_t1, self.Zt1gt3)
                                + torch.matmul(g_t2, self.Zt2gt3) + torch.matmul(g_t3, self.Zt3gt3)
                                + torch.matmul(g_s_spine, self.Zssgt3) + torch.matmul(g_s_left_arm, self.Zslagt3)
                                + torch.matmul(g_s_right_arm, self.Zsragt3) + torch.matmul(g_s_left_leg, self.Zsllgt3)
                                + torch.matmul(g_s_right_leg, self.Zsrlgt3) + self.bgt3)
        f_gss_n = torch.sigmoid(torch.matmul(p, self.Ugss) + torch.matmul(h_t_before_after, self.Wtgss)
                                + torch.matmul(h_s_before_after, self.Wsgss) + torch.matmul(g_t, self.Ztgss)
                                + torch.matmul(g_s, self.Zsgss) + torch.matmul(g_t1, self.Zt1gss)
                                + torch.matmul(g_t2, self.Zt2gss) + torch.matmul(g_t3, self.Zt3gss)
                                + torch.matmul(g_s_spine, self.Zssgss) + torch.matmul(g_s_left_arm, self.Zslagss)
                                + torch.matmul(g_s_right_arm, self.Zsragss) + torch.matmul(g_s_left_leg, self.Zsllgss)
                                + torch.matmul(g_s_right_leg, self.Zsrlgss) + self.bgss)
        f_gsla_n = torch.sigmoid(torch.matmul(p, self.Ugsla) + torch.matmul(h_t_before_after, self.Wtgsla)
                                + torch.matmul(h_s_before_after, self.Wsgsla) + torch.matmul(g_t, self.Ztgsla)
                                + torch.matmul(g_s, self.Zsgsla) + torch.matmul(g_t1, self.Zt1gsla)
                                + torch.matmul(g_t2, self.Zt2gsla) + torch.matmul(g_t3, self.Zt3gsla)
                                + torch.matmul(g_s_spine, self.Zssgsla) + torch.matmul(g_s_left_arm, self.Zslagsla)
                                + torch.matmul(g_s_right_arm, self.Zsragsla) + torch.matmul(g_s_left_leg, self.Zsllgsla)
                                + torch.matmul(g_s_right_leg, self.Zsrlgsla) + self.bgsla)
        f_gsra_n = torch.sigmoid(torch.matmul(p, self.Ugsra) + torch.matmul(h_t_before_after, self.Wtgsra)
                                + torch.matmul(h_s_before_after, self.Wsgsra) + torch.matmul(g_t, self.Ztgsra)
                                + torch.matmul(g_s, self.Zsgsra) + torch.matmul(g_t1, self.Zt1gsra)
                                + torch.matmul(g_t2, self.Zt2gsra) + torch.matmul(g_t3, self.Zt3gsra)
                                + torch.matmul(g_s_spine, self.Zssgsra) + torch.matmul(g_s_left_arm, self.Zslagsra)
                                + torch.matmul(g_s_right_arm, self.Zsragsra) + torch.matmul(g_s_left_leg, self.Zsllgsra)
                                + torch.matmul(g_s_right_leg, self.Zsrlgsra) + self.bgsra)
        f_gsll_n = torch.sigmoid(torch.matmul(p, self.Ugsll) + torch.matmul(h_t_before_after, self.Wtgsll)
                                + torch.matmul(h_s_before_after, self.Wsgsll) + torch.matmul(g_t, self.Ztgsll)
                                + torch.matmul(g_s, self.Zsgsll) + torch.matmul(g_t1, self.Zt1gsll)
                                + torch.matmul(g_t2, self.Zt2gsll) + torch.matmul(g_t3, self.Zt3gsll)
                                + torch.matmul(g_s_spine, self.Zssgsll) + torch.matmul(g_s_left_arm, self.Zslagsll)
                                + torch.matmul(g_s_right_arm, self.Zsragsll) + torch.matmul(g_s_left_leg, self.Zsllgsll)
                                + torch.matmul(g_s_right_leg, self.Zsrlgsll) + self.bgsll)
        f_gsrl_n = torch.sigmoid(torch.matmul(p, self.Ugsrl) + torch.matmul(h_t_before_after, self.Wtgsrl)
                                + torch.matmul(h_s_before_after, self.Wsgsrl) + torch.matmul(g_t, self.Ztgsrl)
                                + torch.matmul(g_s, self.Zsgsrl) + torch.matmul(g_t1, self.Zt1gsrl)
                                + torch.matmul(g_t2, self.Zt2gsrl) + torch.matmul(g_t3, self.Zt3gsrl)
                                + torch.matmul(g_s_spine, self.Zssgsrl) + torch.matmul(g_s_left_arm, self.Zslagsrl)
                                + torch.matmul(g_s_right_arm, self.Zsragsrl) + torch.matmul(g_s_left_leg, self.Zsllgsrl)
                                + torch.matmul(g_s_right_leg, self.Zsrlgsrl) + self.bgsrl)




        # 输出门
        o_n = torch.sigmoid(torch.matmul(p, self.Uo) + torch.matmul(h_t_before_after, self.Wto)
                                 + torch.matmul(h_s_before_after, self.Wso) + torch.matmul(g_t, self.Zto)
                                 + torch.matmul(g_s, self.Zso) + torch.matmul(g_t1, self.Zt1o)
                                 + torch.matmul(g_t2, self.Zt2o) + torch.matmul(g_t3, self.Zt3o)
                                 + torch.matmul(g_s_spine, self.Zsso) + torch.matmul(g_s_left_arm, self.Zslao)
                                 + torch.matmul(g_s_right_arm, self.Zsrao) + torch.matmul(g_s_left_leg,
                                                                                             self.Zsllo)
                                 + torch.matmul(g_s_right_leg, self.Zsrlo) + self.bo)

        # 候选细胞状态
        # c_n = torch.tanh(torch.matmul(p, self.Uc) + torch.matmul(h_t_before_after, self.Wtc)
        #                  + torch.matmul(h_s_before, self.Wsc) + torch.matmul(g_t, self.Ztc)
        #                  + torch.matmul(g_s, self.Zsc) + self.bc)

        # 候选细胞状态
        c_n = torch.tanh(torch.matmul(p, self.Uc) + torch.matmul(h_t_before_after, self.Wtc)
                            + torch.matmul(h_s_before_after, self.Wsc) + torch.matmul(g_t, self.Ztc)
                            + torch.matmul(g_s, self.Zsc) + torch.matmul(g_t1, self.Zt1c)
                            + torch.matmul(g_t2, self.Zt2c) + torch.matmul(g_t3, self.Zt3c)
                            + torch.matmul(g_s_spine, self.Zssc) + torch.matmul(g_s_left_arm, self.Zslac)
                            + torch.matmul(g_s_right_arm, self.Zsrac) + torch.matmul(g_s_left_leg,
                                                                                     self.Zsllc)
                            + torch.matmul(g_s_right_leg, self.Zsrlc) + self.bc)


        # 细胞状态
        # 增加 f_gt1_n, f_gt2_n, f_gt3_n, f_gss_n, f_gsla_n, f_gsra_n, f_gsll_n, f_gsrl_n
        # 对应 c_g_t1, c_g_t2,c_g_t3， c_g_s_spine, c_g_s_left_arm, c_g_s_right_arm,c_g_s_left_leg,c_g_s_right_leg
        # c_h = (f_lt_n * c_t_before) + (f_ft_n * c_h) + (f_rt_n * c_t_after) + (f_s_n * c_s_before)\
        #                 + (f_gt_n * c_g_t) + (f_gs_n * c_g_s) + (c_n * i_n)
        c_h = (f_lt_n * c_t_before) + (f_ft_n * c_h) + (f_rt_n * c_t_after) + (f_s_n * c_h)\
                        + (f_ls_n * c_s_before) + (f_rs_n * c_s_after) \
                        + (f_gt_n * c_g_t) + (f_gs_n * c_g_s) + (c_n * i_n)\
                        + (f_gt1_n * c_g_t1) + (f_gt2_n * c_g_t2) + (f_gt3_n * c_g_t3)\
                        + (f_gss_n * c_g_s_spine) + (f_gsla_n * c_g_s_left_arm) + (f_gsra_n * c_g_s_right_arm)\
                        + (f_gsll_n * c_g_s_left_leg) + (f_gsrl_n * c_g_s_right_leg)

        h = o_n * torch.tanh(c_h)

        c_h = F.dropout(c_h, self.config.keep_prob, train)
        h = F.dropout(h, self.config.keep_prob, train)
        """Update g_t"""
        g_t_hat = torch.mean(h, 1, keepdim=True).expand_as(h)
        f_gtf_n = torch.sigmoid(torch.matmul(g_t, self.Wgtf) + torch.matmul(g_t_hat, self.Zgtf) + self.bgtf)
        f_gtg_n = torch.sigmoid(torch.matmul(g_t, self.Wgtg) + torch.matmul(g_t_hat, self.Zgtg) + self.bgtg)
        o_gt_n = torch.sigmoid(torch.matmul(g_t, self.Wgto) + torch.matmul(g_t_hat, self.Zgto) + self.bgto)

        c_g_t = torch.sum(f_gtf_n * c_h, dim=1, keepdim=True).expand_as(c_h) + c_g_t * f_gtg_n
        g_t = o_gt_n * torch.tanh(c_g_t)


        """Update g_t1"""
        # h: hidden states of [batch, input_window_size-1, nbones, hidden_size]
        # gt1: (self.config.input_window_size - 1)//3
        g_t1_hat = torch.mean(h[:, :(self.config.input_window_size - 1)//3, :, :], 1, keepdim=True).expand_as(h)
        f_gt1f_n = torch.sigmoid(torch.matmul(g_t1, self.Wgt1f) + torch.matmul(g_t1_hat, self.Zgt1f) + self.bgt1f)
        f_gt1g_n = torch.sigmoid(torch.matmul(g_t1, self.Wgt1g) + torch.matmul(g_t1_hat, self.Zgt1g) + self.bgt1g)
        o_gt1_n = torch.sigmoid(torch.matmul(g_t1, self.Wgt1o) + torch.matmul(g_t1_hat, self.Zgt1o) + self.bgt1o)

        c_g_t1 = torch.sum(f_gt1f_n * c_h, dim=1, keepdim=True).expand_as(c_h) + c_g_t1 * f_gt1g_n
        g_t1 = o_gt1_n * torch.tanh(c_g_t1)

        """Update g_t2"""
        # h: hidden states of [batch, input_window_size-1, nbones, hidden_size]
        # gt2: (self.config.input_window_size - 1)//3
        g_t2_hat = torch.mean(h[:, (self.config.input_window_size - 1)//3 : 2*(self.config.input_window_size - 1)//3, :, :], 1, keepdim=True).expand_as(h)
        f_gt2f_n = torch.sigmoid(torch.matmul(g_t2, self.Wgt2f) + torch.matmul(g_t2_hat, self.Zgt2f) + self.bgt2f)
        f_gt2g_n = torch.sigmoid(torch.matmul(g_t2, self.Wgt2g) + torch.matmul(g_t2_hat, self.Zgt2g) + self.bgt2g)
        o_gt2_n = torch.sigmoid(torch.matmul(g_t2, self.Wgt2o) + torch.matmul(g_t2_hat, self.Zgt2o) + self.bgt2o)

        c_g_t2 = torch.sum(f_gt2f_n * c_h, dim=1, keepdim=True).expand_as(c_h) \
                 + c_g_t2 * f_gt2g_n
        g_t2 = o_gt2_n * torch.tanh(c_g_t2)

        """Update g_t3"""
        # h: hidden states of [batch, input_window_size-1, nbones, hidden_size]
        g_t3_hat = torch.mean(h[:, 2*(self.config.input_window_size - 1)//3 :, :, :], 1, keepdim=True).expand_as(h)
        f_gt3f_n = torch.sigmoid(torch.matmul(g_t3, self.Wgt3f) + torch.matmul(g_t3_hat, self.Zgt3f) + self.bgt3f)
        f_gt3g_n = torch.sigmoid(torch.matmul(g_t3, self.Wgt3g) + torch.matmul(g_t3_hat, self.Zgt3g) + self.bgt3g)
        o_gt3_n = torch.sigmoid(torch.matmul(g_t3, self.Wgt3o) + torch.matmul(g_t3_hat, self.Zgt3o) + self.bgt3o)

        c_g_t3 = torch.sum(f_gt3f_n * c_h, dim=1, keepdim=True).expand_as(c_h) \
                 + c_g_t3 * f_gt3g_n
        g_t3 = o_gt3_n * torch.tanh(c_g_t3)

        """Update g_s"""
        g_s_hat = torch.mean(h, 2, keepdim=True).expand_as(h)
        f_gsf_n = torch.sigmoid(torch.matmul(g_s, self.Wgsf) + torch.matmul(g_s_hat, self.Zgsf) + self.bgsf)
        f_gsg_n = torch.sigmoid(torch.matmul(g_s, self.Wgsg) + torch.matmul(g_s_hat, self.Zgsg) + self.bgsg)
        o_gs_n = torch.sigmoid(torch.matmul(g_s, self.Wgso) + torch.matmul(g_s_hat, self.Zgso) + self.bgso)

        c_g_s = torch.sum(f_gsf_n * c_h, dim=2, keepdim=True).expand_as(c_h) + c_g_s * f_gsg_n
        g_s = o_gs_n * torch.tanh(c_g_s)

        """Update g_s_spine"""
        g_ss_hat = torch.mean(h[:, :, self.config.spine_id, :], 2, keepdim=True).expand_as(h)
        f_gssf_n = torch.sigmoid(torch.matmul(g_s_spine, self.Wgssf) + torch.matmul(g_ss_hat, self.Zgssf) + self.bgssf)
        f_gssg_n = torch.sigmoid(torch.matmul(g_s_spine, self.Wgssg) + torch.matmul(g_ss_hat, self.Zgssg) + self.bgssg)
        o_gss_n = torch.sigmoid(torch.matmul(g_s_spine, self.Wgsso) + torch.matmul(g_ss_hat, self.Zgsso) + self.bgsso)

        c_g_s_spine = torch.sum(f_gssf_n * c_h, dim=2, keepdim=True).expand_as(c_h) + c_g_s_spine * f_gssg_n
        g_s_spine = o_gss_n * torch.tanh(c_g_s_spine)

        # # 结果是[batch, input_window_size-1, hidden_size]，在空间维（骨骼）求了平均
        # g_s_spine = torch.mean(h[:, 6:10, :, :], 1, keepdim = True).expand_as(h[:, 6:10, :, :])
        # g_s_left_arm = torch.mean(h[:, 10:14, :, :], 1, keepdim = True).expand_as(h[:, 10:14, :, :])
        # g_s_right_arm = torch.mean(h[:, 14:, :, :], 1, keepdim=True).expand_as(h[:, 14:, :, :])
        # g_s_left_leg = torch.mean(h[:, 3:6, :, :], 1, keepdim=True).expand_as(h[:, 3:6, :, :])
        # g_s_right_leg = torch.mean(h[:, :3, :, :], 1, keepdim=True).expand_as(h[:, :3, :, :])
        """Update g_s_left_arm"""
        g_sla_hat = torch.mean(h[:, :, self.config.left_arm_id, :], 2, keepdim=True).expand_as(h)
        f_gslaf_n = torch.sigmoid(torch.matmul(g_s_left_arm, self.Wgslaf) + torch.matmul(g_sla_hat, self.Zgslaf) + self.bgslaf)
        f_gslag_n = torch.sigmoid(torch.matmul(g_s_left_arm, self.Wgslag) + torch.matmul(g_sla_hat, self.Zgslag) + self.bgslag)
        o_gsla_n = torch.sigmoid(torch.matmul(g_s_left_arm, self.Wgslao) + torch.matmul(g_sla_hat, self.Zgslao) + self.bgslao)

        c_g_s_left_arm = torch.sum(f_gslaf_n * c_h, dim=2, keepdim=True).expand_as(c_h) + c_g_s_left_arm * f_gslag_n
        g_s_left_arm = o_gsla_n * torch.tanh(c_g_s_left_arm)

        """Update g_s_right_arm"""
        g_sra_hat = torch.mean(h[:, :, self.config.right_arm_id, :], 2, keepdim=True).expand_as(h)
        f_gsraf_n = torch.sigmoid(torch.matmul(g_s_right_arm, self.Wgsraf) + torch.matmul(g_sra_hat, self.Zgsraf) + self.bgsraf)
        f_gsrag_n = torch.sigmoid(torch.matmul(g_s_right_arm, self.Wgsrag) + torch.matmul(g_sra_hat, self.Zgsrag) + self.bgsrag)
        o_gsra_n = torch.sigmoid(torch.matmul(g_s_right_arm, self.Wgsrao) + torch.matmul(g_sra_hat, self.Zgsrao) + self.bgsrao)

        c_g_s_right_arm = torch.sum(f_gsraf_n * c_h, dim=2, keepdim=True).expand_as(c_h) + c_g_s_right_arm * f_gsrag_n
        g_s_right_arm = o_gsra_n * torch.tanh(c_g_s_right_arm)

        """Update g_s_left_leg"""
        g_sll_hat = torch.mean(h[:, :, self.config.left_leg_id, :], 2, keepdim=True).expand_as(h)
        f_gsllf_n = torch.sigmoid(torch.matmul(g_s_left_leg, self.Wgsllf) + torch.matmul(g_sll_hat, self.Zgsllf) + self.bgsllf)
        f_gsllg_n = torch.sigmoid(torch.matmul(g_s_left_leg, self.Wgsllg) + torch.matmul(g_sll_hat, self.Zgsllg) + self.bgsllg)
        o_gsll_n = torch.sigmoid(torch.matmul(g_s_left_leg, self.Wgsllo) + torch.matmul(g_sll_hat, self.Zgsllo) + self.bgsllo)

        c_g_s_left_leg = torch.sum(f_gsllf_n * c_h, dim=2, keepdim=True).expand_as(c_h) + c_g_s_left_leg * f_gsllg_n
        g_s_left_leg = o_gsll_n * torch.tanh(c_g_s_left_leg)

        """Update g_s_right_leg"""
        g_srl_hat = torch.mean(h[:, :, self.config.right_leg_id, :], 2, keepdim=True).expand_as(h)
        f_gsrlf_n = torch.sigmoid(torch.matmul(g_s_right_leg, self.Wgsrlf) + torch.matmul(g_srl_hat, self.Zgsrlf) + self.bgsrlf)
        f_gsrlg_n = torch.sigmoid(torch.matmul(g_s_right_leg, self.Wgsrlg) + torch.matmul(g_srl_hat, self.Zgsrlg) + self.bgsrlg)
        o_gsrl_n = torch.sigmoid(torch.matmul(g_s_right_leg, self.Wgsrlo) + torch.matmul(g_srl_hat, self.Zgsrlo) + self.bgsrlo)

        c_g_s_right_leg = torch.sum(f_gsrlf_n * c_h, dim=2, keepdim=True).expand_as(c_h) + c_g_s_right_leg * f_gsrlg_n
        g_s_right_leg = o_gsrl_n * torch.tanh(c_g_s_right_leg)


        hidden_states = h.view([h.shape[0], h.shape[1], -1])
        cell_states = c_h.view([c_h.shape[0], c_h.shape[1], -1])
        global_t_state = g_t[:, 1, :, :].view([g_t.shape[0], -1])
        return hidden_states, cell_states, global_t_state, g_t, c_g_t, g_t1, g_t2, g_t3,c_g_t1, c_g_t2,c_g_t3,g_s, c_g_s,g_s_spine, c_g_s_spine,g_s_left_arm,c_g_s_left_arm, g_s_right_arm,c_g_s_right_arm,g_s_left_leg,c_g_s_left_leg,g_s_right_leg,c_g_s_right_leg


class Kinematics_LSTM_decoder(nn.Module):

    def __init__(self, config):
        """
        This decoder only apply to h3.6m dataset.
        :param config: global config class
        """
        super().__init__()
        self.config = config
        self.seq_length_out = config.output_window_size
        self.nbones = config.nbones
        self.lstm = nn.ModuleList()
        self.para_list = torch.nn.ParameterList()
        if config.dataset == 'Human':
            co = 5
        for i in range(co):
            self.para_list.append(torch.nn.Parameter(torch.empty(int(config.input_size/config.bone_dim*config.hidden_size), config.training_chain_length[i]).uniform_(-0.04, 0.04)))
            self.para_list.append(torch.nn.Parameter(torch.empty(config.training_chain_length[i]).uniform_(-0.04, 0.04)))
        # LSTM First layer
        self.lstm.append(nn.LSTMCell(config.input_size, int(config.input_size / config.bone_dim * config.hidden_size)))
        # Kinematics LSTM layer
        spine = nn.LSTMCell(int(config.input_size / config.bone_dim * config.hidden_size), int(config.input_size / config.bone_dim * config.hidden_size))
        self.lstm.append(spine)
        arm = nn.LSTMCell(int(config.input_size / config.bone_dim * config.hidden_size), int(config.input_size / config.bone_dim * config.hidden_size))
        self.lstm.append(arm)
        self.lstm.append(arm)
        if config.dataset == 'Human':
            leg = nn.LSTMCell(int(config.input_size / config.bone_dim * config.hidden_size), int(config.input_size / config.bone_dim * config.hidden_size))
            self.lstm.append(leg)
            self.lstm.append(leg)
            self.lstm_layer = 6
        elif config.dataset == 'CSL':
            self.lstm_layer = 4

    def forward(self, hidden_states, cell_states, global_t_state, p):
        """
        decoder forward
        :param hidden_states: hideen states [batch, input_window_size-1, nbones * hidden_size]
        :param cell_states: hideen states [batch, input_window_size-1, nbones * hidden_size]
        :param global_t_state: [batch, nbones * hidden_size]
        :param p: [batch, 1, nbones * hidden_size] 1 remains the dimension of hidden states
        :return: predictions of human motion
        """

        # define decoder hidden states and cell states
        h = []
        c_h = []
        pre = torch.zeros([hidden_states.shape[0], self.seq_length_out, self.config.input_size], device=p.device)
        for i in range(self.lstm_layer):
            h.append(torch.zeros(hidden_states.shape[0], self.seq_length_out + 1, self.nbones * self.config.hidden_size,
                              device=p.device))
            c_h.append(torch.zeros_like(h[i]))
            # feed init hidden states and cell states into h and c_h
            if i == 0:
                h_t = hidden_states
            elif i == 1:
                h_t = torch.cat((global_t_state.unsqueeze(1), hidden_states), dim=1)
            if i < 2:
                h[i][:, 0, :] = h_t.mean(dim=1)
                c_h[i][:, 0, :] = torch.mean(cell_states, dim=1)

        for frame in range(self.seq_length_out):
            for i in range(self.lstm_layer):
                # 0 全局层；1 脊柱层； 2 3 左右手臂层； 4 5 左右腿部层
                cell = self.lstm[i]
                if i == 0:
                    if frame == 0:
                        input = p[:, 0, :]
                        input_first = p[:, 0, :]
                    else:
                        input = pre[:, frame - 1, :].clone()
                        input_first = pre[:, frame - 1, :].clone()
                else:
                    if i == (3 or 4 or 5):
                        input = h[1][:, frame + 1, :].clone()
                    else:
                        input = h[i-1][:, frame + 1, :].clone()
                h[i][:, frame + 1, :], c_h[i][:, frame + 1, :] \
                    = cell(input, (h[i][:, frame, :].clone(), c_h[i][:, frame, :].clone()))
                if self.config.dataset == 'Human':
                    order = [2, 0, 1, 3, 4]
                elif self.config.dataset == 'CSL':
                    order = [0, 1, 2]
                if i != 0:
                    pre[:, frame, self.config.index[order[i-1]]] = torch.matmul(h[i][:, frame + 1, :].clone(),
                                                                    self.para_list[order[i-1]*2]) + self.para_list[order[i-1]*2+1] + input_first[:, self.config.index[order[i-1]]]
                """"""
                # if i == 1:
                #     pre[:, frame, self.config.index[2]] = torch.matmul(h[i][:, frame + 1, :].clone(), self.weights_out_spine) + \
                #                    self.bias_out_spine + input_first[:, self.config.index[2]]
                # elif i == 2:
                #     pre[:, frame, self.config.index[0]] = torch.matmul(h[i][:, frame + 1, :].clone(), self.weights_out_leg1) + \
                #                    self.bias_out_leg1 + input_first[:, self.config.index[0]]
                # elif i == 3:
                #     pre[:, frame, self.config.index[1]] = torch.matmul(h[i][:, frame + 1, :].clone(), self.weights_out_leg2) + \
                #                    self.bias_out_leg2 + input_first[:, self.config.index[1]]
                # elif i == 4:
                #     pre[:, frame, self.config.index[3]] = torch.matmul(h[i][:, frame + 1, :].clone(), self.weights_out_arm1) + \
                #                    self.bias_out_arm1 + input_first[:, self.config.index[3]]
                # elif i == 5:
                #     pre[:, frame, self.config.index[4]] = torch.matmul(h[i][:, frame + 1, :].clone(), self.weights_out_arm2) + \
                #                    self.bias_out_arm2 + input_first[:, self.config.index[4]]
                """"""
                # if i == 1:
                #     pre[:, frame, self.config.index[0]] = torch.matmul(h[i][:, frame + 1, :].clone(), self.weights_out_spine) + \
                #                    self.bias_out_spine + input_first[:, self.config.index[0]]
                # elif i == 2:
                #     pre[:, frame, self.config.index[1]] = torch.matmul(h[i][:, frame + 1, :].clone(), self.weights_out_arm1) + \
                #                    self.bias_out_arm1 + input_first[:, self.config.index[1]]
                # elif i == 3:
                #     pre[:, frame, self.config.index[2]] = torch.matmul(h[i][:, frame + 1, :].clone(), self.weights_out_arm2) + \
                #                    self.bias_out_arm2 + input_first[:, self.config.index[2]]
                """"""""

        return pre


class LSTM_decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_length_out = config.output_window_size
        self.nbones = config.nbones
        self.lstm = nn.ModuleList()
        self.weights_out = torch.nn.Parameter(torch.empty(int(config.input_size/config.bone_dim*config.hidden_size), config.input_size).uniform_(-0.04, 0.04))
        self.bias_out = torch.nn.Parameter(torch.empty(config.input_size).uniform_(-0.04, 0.04))
        for i in range(config.decoder_recurrent_steps):
            if i == 0:
                self.lstm.append(nn.LSTMCell(config.input_size, int(config.input_size/config.bone_dim*config.hidden_size)))
            else:
                self.lstm.append(nn.LSTMCell(int(config.input_size/config.bone_dim*config.hidden_size), int(config.input_size/config.bone_dim*config.hidden_size)))

    def forward(self, hidden_states, cell_states, global_t_state, p):
        """
        decoder forward
        :param hidden_states: hideen states [batch, input_window_size-1, nbones * hidden_size]
        :param cell_states: hideen states [batch, input_window_size-1, nbones * hidden_size]
        :param global_t_state: [batch, nbones * hidden_size]
        :param p: [batch, 1, nbones * hidden_size] 1 remains the dimension of hidden states
        :return: predictions of human motion
        """

        # define decoder hidden states and cell states
        h = []
        c_h = []
        pre = torch.zeros([hidden_states.shape[0], self.seq_length_out, self.config.input_size], device=p.device)
        for i in range(self.config.decoder_recurrent_steps):
            h.append(torch.zeros(hidden_states.shape[0], self.seq_length_out + 1, self.nbones * self.config.hidden_size,
                              device=p.device))
            c_h.append(torch.zeros_like(h[i]))
            # feed init hidden states and cell states into h and c_h
            if i == 0:
                h_t = hidden_states
            elif i == 1:
                h_t = torch.cat((global_t_state.unsqueeze(1), hidden_states), dim=1)
            else:
                raise Exception('The max decoder_recurrent_steps is 2!')
            h[i][:, 0, :] = h_t.mean(dim=1)
            c_h[i][:, 0, :] = torch.mean(cell_states, dim=1)

        for frame in range(self.seq_length_out):
            for i in range(self.config.decoder_recurrent_steps):
                cell = self.lstm[i]
                if i == 0:
                    if frame == 0:
                        input = p[:, 0, :]
                        input_first = p[:, 0, :]
                    else:
                        input = pre[:, frame - 1, :].clone()
                        input_first = pre[:, frame - 1, :].clone()
                else:
                    input = h[i - 1][:, frame + 1, :].clone()
                h[i][:, frame + 1, :], c_h[i][:, frame + 1, :] \
                    = cell(input, (h[i][:, frame, :].clone(), c_h[i][:, frame, :].clone()))
            pre[:, frame, :] = torch.matmul(h[-1][:, frame + 1, :].clone(), self.weights_out) + \
                           self.bias_out + input_first
        return pre


class ST_LSTM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config = config.nbones
        recurrent_cell_box = torch.nn.ModuleList()
        self.seq_length_out = config.output_window_size
        self.weights_out = torch.nn.Parameter(torch.empty(config.hidden_size, config.bone_dim).uniform_(-0.04, 0.04))
        self.bias_out = torch.nn.Parameter(torch.empty(config.bone_dim).uniform_(-0.04, 0.04))

        for i in range(config.decoder_recurrent_steps):
            cells = torch.nn.ModuleList()
            for bone in range(config.nbones):
                if i == 0:
                    cell = ST_LSTMCell(config.bone_dim, config.hidden_size)
                else:
                    cell = ST_LSTMCell(config.hidden_size, config.hidden_size)
                cells.append(cell)
            recurrent_cell_box.append(cells)
        self.recurrent_cell_box = recurrent_cell_box

    def forward(self, hidden_states, cell_states, global_t_state, global_s_state, p):
        """
        :param hidden_states:  [batch, input_window_size-1, nbones, hidden_size]
        :param cell_states: [batch, input_window_size-1, nbones, hidden_size]
        :param global_t_state: [batch,  nbones, hidden_size]
        :param global_s_state: [batch, input_window_size-1, hidden_size]
        :param p: [batch, input_window_size-1, nbones, hidden_size]
        :return:
        """

        # define decoder hidden states and cell states
        h = []
        c_h = []
        pre = torch.zeros([hidden_states.shape[0], self.seq_length_out, self.nbones, self.config.bone_dim], device=p.device)
        p = p.view([p.shape[0], p.shape[1], self.nbones, self.config.bone_dim])
        for i in range(self.config.decoder_recurrent_steps):
            h.append(torch.zeros(hidden_states.shape[0], self.seq_length_out + 1, self.nbones + 1, self.config.hidden_size,
                              device=p.device))
            c_h.append(torch.zeros_like(h[i]))
            # feed init hidden states and cell states into h and c_h
            if i == 0:
                h_t = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], self.nbones, self.config.hidden_size)
            elif i == 1:
                h_t = torch.cat((global_t_state.unsqueeze(1), hidden_states), dim=1).view(hidden_states.shape[0], hidden_states.shape[1]+1, self.nbones, self.config.hidden_size)
            else:
                print('The max decoder num is 2!')

            h[i][:, 0, 1:, :] = torch.mean(h_t, dim=1)
            c_h[i][:, 0, 1:, :] = torch.mean(cell_states.view(cell_states.shape[0], cell_states.shape[1], self.nbones, self.config.hidden_size), dim=1)

        for frame in range(self.seq_length_out):
            for i in range(self.config.decoder_recurrent_steps):
                for bone in range(self.nbones):
                    cell = self.recurrent_cell_box[i][bone]
                    if i == 0:
                        if frame == 0:
                            input = p[:, 0, bone, :]
                            input_first = p[:, 0, bone, :]
                        else:
                            input = pre[:, frame - 1, bone, :].clone()
                            input_first = pre[:, frame - 1, bone, :].clone()
                    else:
                        input = h[i - 1][:, frame + 1, bone, :].clone()

                    h[i][:, frame+1, bone+1, :], c_h[i][:, frame+1, bone+1, :] \
                        = cell(input, h[i][:, frame, bone+1, :].clone(),
                            h[i][:, frame+1, bone, :].clone(), c_h[i][:, frame, bone+1, :].clone(), c_h[i][:, frame+1, bone, :].clone())
            pre[:, frame, :, :] = torch.matmul(h[-1][:, frame + 1, :, :].clone(), self.weights_out) + self.bias_out + input_first
        pre_c = c_h[-1][:, 1:, 1:, :].view([c_h[-1][:, 1:, 1:, :].shape[0], c_h[-1][:, 1:, 1:, :].shape[1], -1])
        pre = pre.view([pre.shape[0], pre.shape[1], -1])

        return pre, pre_c


class ST_LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()

        # input gate
        self.Ui = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.Wti = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Wsi = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bi = torch.nn.Parameter(torch.randn(hidden_size))
        # space forget gate
        self.Us = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.Wts = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Wss = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bs = torch.nn.Parameter(torch.randn(hidden_size))
        # time forget gate
        self.Ut = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.Wtt = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Wst = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bt = torch.nn.Parameter(torch.randn(hidden_size))
        # output gate
        self.Uo = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.Wto = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Wso = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bo = torch.nn.Parameter(torch.randn(hidden_size))
        # c_hat gate
        self.Uc = torch.nn.Parameter(torch.randn(input_size, hidden_size))
        self.Wtc = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.Wsc = torch.nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.bc = torch.nn.Parameter(torch.randn(hidden_size))

    def forward(self, x, h_t, h_s, c_t, c_s):

        i_n = torch.sigmoid(torch.matmul(x, self.Ui) + torch.matmul(h_t, self.Wti) + torch.matmul(h_s, self.Wsi) + self.bi)
        f_s_n = torch.sigmoid(torch.matmul(x, self.Us) + torch.matmul(h_t, self.Wts) + torch.matmul(h_s, self.Wss) + self.bs)
        f_t_n = torch.sigmoid(torch.matmul(x, self.Ut) + torch.matmul(h_t, self.Wtt) + torch.matmul(h_s, self.Wst) + self.bt)
        o_n = torch.sigmoid(torch.matmul(x, self.Uo) + torch.matmul(h_t, self.Wto) + torch.matmul(h_s, self.Wso) + self.bo)
        c_n = torch.tanh(torch.matmul(x, self.Uc) + torch.matmul(h_t, self.Wtc) + torch.matmul(h_s, self.Wsc) + self.bc)

        c_h = (i_n * c_n) + (f_t_n * c_t) + (f_s_n * c_s)
        h = o_n * torch.tanh(c_h)

        return h, c_h

