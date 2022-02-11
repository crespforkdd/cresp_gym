import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .agent_base import AGENTBase
from common import utils
from module.make_module import init_algo
from itertools import chain
from collections import deque


# CRESP to predict Characteristic Functions 
# with given action sequences
class CRESPAgent(AGENTBase):

    def __init__(self, aug_func, critic_lr, critic_beta, actor_lr, actor_beta,
                 actor_mode, extr_lr, alpha_lr, alpha_beta, extr_q_update_freq,
                 cf_config, rs_discount, temperature, config, **kwargs):
        super().__init__(device=config['device'], **config['agent_base_params'])
        # Setting hyperparameters
        obs_shape = config['obs_shape']
        self.batch_size = config['batch_size']
        self.actor_mode = actor_mode
        self.extr_q_update_freq = extr_q_update_freq if extr_q_update_freq is not None else 0
        self.cf_config = cf_config
        self.pred_temp = temperature
        self.rs_discount = rs_discount
        r, k = cf_config.split('-')
        self.r_dim = int(r.split('r')[-1]) if 'r' in r else 1
        self.k_dim = int(k.split('k')[-1]) if 'k' in k else 64
        self.nsr = config['algo_params']['nstep_rew']
        config['algo_params']['r_dim'] = self.r_dim
        config['algo_params']['k_dim'] = self.k_dim

        # Setting modules
        self._init_extractor(obs_shape, config['extr_params'])
        repr_dim = self.extr.repr_dim
        self.rl = init_algo('sac', (self.extr.is_fc, repr_dim, repr_dim), config['algo_params'])
        # Setting augmentation
        self.aug_func = aug_func

        # Setting optimizers
        self.extr_optimizer = torch.optim.Adam(self.extr.parameters(), lr=extr_lr)
        self.critic_optimizer = torch.optim.Adam(self.rl.critic.parameters(),
                                                 lr=critic_lr, betas=(critic_beta, 0.999))
        self.actor_optimizer  = torch.optim.Adam(self.rl.actor.parameters(),
                                                 lr=actor_lr, betas=(actor_beta, 0.999))
        self.log_alpha_optimizer = torch.optim.Adam([self.rl.log_alpha],
                                                     lr=alpha_lr, betas=(alpha_beta, 0.999))
        self.proj_optimizer = torch.optim.Adam(self.rl.actor.parameters(), lr=extr_lr)

        self.cl_loss = nn.CrossEntropyLoss().to(self.device)
        self.debug_info['mat'], self.debug_info['sm'] = {}, {}

        self.train()
        self.train_targ()
        self.print_module()

    def _handle_data(self, data):
        if (data > 1.0).any():
            return data / 255.0# - 0.5
        return data

    def _print_module(self):
        pass

    @torch.no_grad()
    def select_action(self, o, deterministic=False, tanh=True):
        o = torch.FloatTensor(o).to(self.device).unsqueeze(0)
        o = self._handle_data(o)
        s = self.extr(o)
        return self.rl.select_action(s, deterministic, tanh)

    def augment(self, data, num=2, name=''):
        o, a, r, next_o = data['obs'], data['act'], data['rew'], data['obs2']
        infos, nd, envls = data['infos'], data['not_done'], data['env_labels']
        r = r.squeeze(-1) if r.size(-1) == 1 else r
        nd = nd.squeeze(-1) if nd.size(-1) == 1 else nd
        self.debug_info['obs%s' % name] = o # (batch_size, *o.size())
        self.debug_info['next_obs%s' % name] = next_o # (batch_size, *next_o.size())
        # Augment data
        batch_size = o.size(0)
        aug_o, aug_next_o = self.aug_func(o.repeat(num, 1, 1, 1)), self.aug_func(next_o.repeat(2, 1, 1, 1))
        aug_o, aug_next_o = self._handle_data(aug_o), self._handle_data(aug_next_o)
        return aug_o, a, r, infos, aug_next_o, nd, envls

    def update_critic(self, o, a, r, infos, next_o, nd, o2, a2, r2, infos2, next_o2, nd2,
                      q_targ, q_targ2, q_targ_info_dict, buffer_infos, L, TBL, save_log):
        utils.freeze_module(self.rl.actor)
        if 'Q' not in self.cf_config and self.extr_q_update_freq == 0 \
                or self.total_time_steps % self.extr_q_update_freq != 0:
            utils.freeze_module(self.extr)

        loss_q, q_info_dict = self.rl.update_critic(self.extr(o), a, num=self.num, q_targ=q_targ)
        loss_q2, _ = self.rl.update_critic(self.extr(o2), a2, num=self.num, q_targ=q_targ2)
        qf_opt_dict = dict(opt_e=self.extr_optimizer, opt_q=self.critic_optimizer)
        utils.update_params(qf_opt_dict, loss_q + loss_q2)
        self.update_critic_steps += 1
        utils.activate_module(self.rl.actor)
        if 'Q' not in self.cf_config and self.extr_q_update_freq == 0 \
                or self.total_time_steps % self.extr_q_update_freq != 0:
            utils.activate_module(self.extr)

        if save_log:
            self.debug_info['critic_log'] = q_info_dict
            self.debug_info['critic_log'].update(buffer_infos)
            q_info_dict.update(q_targ_info_dict)
            self.rl.critic.log(TBL, self.total_time_steps, int(save_log))
            TBL.save_log_param(self.debug_info['critic_log'], 'train_critic_log', self.total_time_steps)
            TBL.save_log(q_info_dict, 'train_critic', self.total_time_steps)
            L.store(**q_info_dict)

    def update_actor(self, o, L, TBL, save_log):
        utils.freeze_module([self.extr, self.rl.critic])
        loss_pi, loss_alpha, pi_info_dict = self.rl.update_actor(self.extr(o).detach(), self.actor_mode, self.num)
        utils.update_params(self.actor_optimizer, loss_pi)
        utils.update_params(self.log_alpha_optimizer, loss_alpha)
        self.update_actor_steps += 1
        utils.activate_module([self.extr, self.rl.critic])

        if save_log:
            self.rl.actor.log(TBL, self.total_time_steps, int(save_log))
            TBL.save_log(pi_info_dict, 'train_actor', self.total_time_steps)
            L.store(**pi_info_dict)

    def output_loss(self, loss1, loss2, mode):
        if 'max' in mode:
            if mode != 'max':
                num = int(mode.split('x')[-1])
                loss_out1 = loss1.sort()[0][-num:].mean()
                loss_out2 = loss2.sort()[0][-num:].mean()
                up_idx1, up_idx2 = loss1.argmax(), loss2.argmax()
            else:
                loss_out1, up_idx1 = loss1.max(0)
                loss_out2, up_idx2 = loss2.max(0)
        elif 'min' in mode:
            if mode != 'min':
                num = int(mode.split('n')[-1])
                loss_out1 = loss1.sort()[0][:num].mean()
                loss_out2 = loss2.sort()[0][:num].mean()
                up_idx1, up_idx2 = loss1.argmin(), loss2.argmin()
            else:
                loss_out1, up_idx1 = loss1.min(0)
                loss_out2, up_idx2 = loss2.min(0)
        return loss_out1 + loss_out2, up_idx1, up_idx2

    def compute_rew_cl_loss(self, pft, ft, pft2, ft2, labels, mode='_max'):
        loss_pft_cl, acc_r, acc_r2 = 0.0, 0.0, 0.0
        mode = mode.split('_')[-1]
        pft = pft.view(pft.size(0), labels.size(0), self.num, -1) # (er_dim, b, n, 2*k_dim)
        pft2 = pft2.view(*pft.size())
        ft = ft.view(labels.size(0), self.num, -1) # (b, n, 2*k_dim)
        ft2 = ft2.view(*ft.size())
        for i in range(self.num):
            loss_cl1, acc1 = utils.compute_cl_loss(pft[:,:,i], ft[:,i], labels, None, self.pred_temp, True)
            loss_cl2, acc2 = utils.compute_cl_loss(pft2[:,:,i], ft2[:,i], labels, None, self.pred_temp, True)
            loss_cl, up_idx_cl1, up_idx_cl2 = self.output_loss(loss_cl1, loss_cl2, mode)
            loss_pft_cl += loss_cl
            acc_r += acc1
            acc_r2 += acc2
        return loss_cl, (acc_r + acc_r2) / 2 / self.num, up_idx_cl1, up_idx_cl2

    def compute_rew_mse_loss(self, pft, ft, pft2, ft2, mode='max'):
        # (er_dim, b*n, 2*k_dim) --> (er_dim,)
        loss_pft_norm = F.mse_loss(pft, ft, reduction='none').mean([1, 2]) * self.num
        loss_pft2_norm = F.mse_loss(pft2, ft2, reduction='none').mean([1, 2]) * self.num
        loss_norm, up_idx_norm1, up_idx_norm2 = self.output_loss(loss_pft_norm, loss_pft2_norm, mode)
        return loss_norm, up_idx_norm1, up_idx_norm2

    def compute_rew_loss(self, data, data2):
        traj_o, traj_a, traj_r, o_end = data['traj_o'], data['traj_a'], data['traj_r'], data['obs_end']
        traj_o2, traj_a2, traj_r2, o2_end = data2['traj_o'], data2['traj_a'], data2['traj_r'], data2['obs_end']
        batch_size = traj_r.size(1)

        t, t2 = self.rl.actor.t(batch_size*self.num), self.rl.actor.t(batch_size*self.num) # (k_dim, b*n, nsr)
        if 'minCF' not in self.cf_config:
            # Do NOT parameterize the distribution W
            t, t2 = t.detach(), t2.detach()
        '''Pretreat Data'''
        with torch.no_grad():
            o = self._handle_data(self.aug_func(traj_o.repeat(self.num, 1, 1, 1)))
            o2 = self._handle_data(self.aug_func(traj_o2.repeat(self.num, 1, 1, 1)))
            a, a2 = traj_a.repeat(1, self.num, 1), traj_a2.repeat(1, self.num, 1) # (nsr, b*n, a_dim)

            '''Calculate Discout Labels for Characteristic Function'''
            r, r2 = traj_r.t().repeat(self.num, 1), traj_r2.t().repeat(self.num, 1) # (b*n, nsr)
            rs_discount = (self.rs_discount ** torch.arange(self.nsr).to(self.device)).view(1, -1)
            dr, dr2 = r * rs_discount, r2 * rs_discount
            cos_t, sin_t = np.pi * torch.cos((t * dr).sum(-1)) / 2, np.pi * torch.sin((t * dr).sum(-1)) / 2 # (k_dim, b*n)
            cos_t2, sin_t2 = np.pi * torch.cos((t2 * dr2).sum(-1)) / 2, np.pi * torch.sin((t2 * dr2).sum(-1)) / 2
            labels = torch.arange(batch_size).long().to(self.device)

            ft = torch.cat([cos_t.T, sin_t.T], dim=-1) # (b*n, 2*k_dim)
            ft2 = torch.cat([cos_t2.T, sin_t2.T], dim=-1)

        '''Predict Characteristic Function'''
        pft = self.rl.actor.pr(self.extr(o), a, t) # (er_dim, b*n, 2*k_dim)
        pft2 = self.rl.actor.pr(self.extr(o2), a2, t2)
        loss_norm, up_idx_norm1, up_idx_norm2 = self.compute_rew_mse_loss(pft, ft, pft2, ft2)
        extr_info_dict = dict(LossNR=loss_norm.clone(), UpN1=float(up_idx_norm1), UpN2=float(up_idx_norm2))
        loss_cl, acc, up_idx_cl1, up_idx_cl2 = self.compute_rew_cl_loss(pft, ft, pft2, ft2, labels)
        extr_info_dict.update(dict(RAcc=acc, LossCLR=loss_cl.clone(), UpCL1=float(up_idx_cl1), UpCL2=float(up_idx_cl2)))
        loss_r = loss_cl + loss_norm
        return loss_r, extr_info_dict

    def update_extr(self, data, data2, L, TBL, save_log):
        utils.freeze_module(self.rl.critic)
        loss_r, extr_info_dict = self.compute_rew_loss(data, data2)

        extr_opt_dict = dict(opt_eq=self.extr_optimizer, opt_pro=self.proj_optimizer)
        utils.update_params(extr_opt_dict, loss_r)
        self.update_extr_steps += 1
        utils.activate_module(self.rl.critic)

        TBL.save_log(extr_info_dict, 'train_extr', self.total_time_steps)
        TBL.save_log_param(extr_info_dict, 'train_extr_log', self.total_time_steps)
        if save_log:
            self.extr.log(TBL, self.total_time_steps, int(save_log))
            for key, values in self.debug_info['sm'].items():
                TBL.log_images('train/%s' % key, values, self.total_time_steps)
            L.store(**extr_info_dict)

    def update(self, replay_buffer, L, TBL, step, save_log, batch_size=None):
        if isinstance(replay_buffer, list):
            replay_buffer, replay_buffer2 = replay_buffer
        else:
            replay_buffer, replay_buffer2 = replay_buffer, replay_buffer
        self.update_steps += 1

        '''Update critic'''
        for i in range(self.update_to_data):
            '''Sample batch from buffer'''
            o, a, r, infos, next_o, nd, _ = self.augment(replay_buffer.sample(batch_size, self), self.num)
            o2, a2, r2, infos2, next_o2, nd2, _ = self.augment(replay_buffer2.sample(batch_size, self), self.num, '2')

            '''Calculate target q'''
            with torch.no_grad():
                q_targ, q_targ_info_dict = self.calculate_target_q(next_o, r, infos, nd)
                q_targ2, _ = self.calculate_target_q(next_o2, r2, infos2, nd2)

            self.update_critic(o, a, r, infos, next_o, nd, o2, a2, r2, infos2, next_o2, nd2,
                               q_targ, q_targ2, q_targ_info_dict, replay_buffer.infos, L, TBL,
                               save_log and (not i))

        '''Update actor'''
        for i in range(self.update_to_actor):
            '''Sample observations from buffer'''
            if i:
                o, a, r, infos, next_o, nd, _ = self.augment(replay_buffer.sample(batch_size, self), self.num)
            self.update_actor(o, L, TBL, save_log and (not i))

        '''Update extr'''
        for i in range(self.update_to_data):
            '''Sample batch from buffer'''
            data = replay_buffer.sample_abst_traj(batch_size, self)
            data2 = replay_buffer2.sample_abst_traj(batch_size, self)
            self.update_extr(data, data2, L, TBL, save_log and (not i))

        '''Smooth update'''
        if step % self.critic_target_update_freq == 0:
            self.rl.soft_update_params()
            if self.extr_targ is not None:
                utils.soft_update_params(self.extr, self.extr_targ, self.extr_tau)

    def _save(self, model_dir, step):
        pass

    def _load(self, model_dir, step):
        pass

    def _print_log(self, logger):
        pass
