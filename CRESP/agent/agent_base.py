import abc
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from module.make_module import init_extractor


class AGENTBase(object, metaclass=abc.ABCMeta):
    
    def __init__(
        self,
        action_repeat,
        actor_update_freq,
        critic_target_update_freq,
        update_to_data,
        update_to_actor,
        num_sources,
        num,
        device
    ):
        # Setting hyperparameters
        self.action_repeat = action_repeat
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.update_to_data = update_to_data
        self.update_to_actor = update_to_actor
        self.num_sources = num_sources
        self.num = num
        self.device = device
        self.debug_info = {}
        self.training = False

        self.total_time_steps = 0
        self.update_steps = 0
        self.update_critic_steps = 0
        self.update_actor_steps = 0
        self.update_extr_steps = 0

        # Setting modules
        self.extr = None
        self.extr_targ = None
        self.rl = None

        # Setting optimizers
        self.extr_q_optimizer = None
        self.critic_optimizer = None
        self.actor_optimizer = None

    def train(self, training=True):
        self.training = training
        self.extr.train(self.training)
        self._train()
        self.rl.train(self.training)

    def _train(self):
        pass

    def train_targ(self):
        self.extr_targ.train(self.training) if self.extr_targ is not None else None
        self._train_targ()
        self.rl.train_targ(self.training)

    def _train_targ(self):
        pass

    def record(self, o, a, r, o2, rec=False):
        pass

    def print_module(self):
        print("Extractor:", self.extr)
        print("Target Extractor:", self.extr_targ)
        print("Critic:", self.rl.critic)
        print("Actor:", self.rl.actor)
        self._print_module()

    def _print_module(self):
        pass

    def _init_extractor(self, obs_shape, extr_config):
        module_dict = init_extractor(obs_shape, self.device, extr_config)
        self.extr = module_dict['extr']
        self.extr_targ = module_dict['extr_targ']
        self.extr_tau = module_dict['extr_tau']

    def select_action(self, o, deterministic=False):
        pass

    @torch.no_grad()
    def calculate_target_q(self, next_o, r, infos, nd, num=2):
        extr_targ = self.extr_targ if self.extr_targ is not None else self.extr
        gamma = infos['gamma']

        s2 = extr_targ(next_o) # (num*batch_size, e_dim)
        q_targ, q_targ_max = self.rl.get_q_target(s2, r.repeat(num), gamma, nd.repeat(num)) # (num*batch_size,)

        q_targ = q_targ.view(num, -1) # (num, batch_size)
        q_targ_max = q_targ_max.view(num, -1) # (num, batch_size)
        q_targ = q_targ.mean(0).unsqueeze(0)

        q_targ_max = q_targ_max.mean(0) # (batch_size,)
        if infos['nstep'] == 1 or infos['lambd'] == -1.0:
            q_targ_info_dict = dict(TQvals=q_targ[0], TQmaxs=q_targ_max)
            return q_targ, q_targ_info_dict

        lambd, traj, rew = infos['lambd'], infos['traj'], infos['rew']
        for i in range(rew.size(0)):
            o1 = self._handle_data(self.aug_func(traj[-i].repeat(num, 1, 1, 1)))
            s1 = extr_targ(o1) # (num*batch_size, e_dim)

            q1, q1_max = self.rl.get_q_target(s1, 0.0, 1.0, 1.0)
            q1, q1_max = q1.view(num, -1), q1_max.view(num, -1).mean(0)
            q1 = q1.mean(0).unsqueeze(0)
            q_targ = rew[-1] + gamma * ((1 - lambd) * q1 + lambd * q_targ)
            q_targ_max = rew[-1] + gamma * ((1 - lambd) * q1_max + lambd * q_targ_max)

        q_targ_info_dict = dict(TQvals=q_targ[0], TQmaxs=q_targ_max)
        return q_targ, q_targ_info_dict

    @abc.abstractmethod
    def update(self, replay_buffer, L, TBL, step, save_log, batch_size=None):
        pass

    def save(self, model_dir, step):
        torch.save(
            self.extr.state_dict(), '%s/extr_%s.pt' % (model_dir, step)
        )
        self.rl.save(model_dir, step)
        self._save(model_dir, step)

    @abc.abstractmethod
    def _save(self, model_dir, step):
        raise NotImplementedError

    def load(self, model_dir, step):
        self.extr.load_state_dict(
            torch.load('%s/extr_%s.pt' % (model_dir, step))#, map_location=torch.device('cpu'))
        )
        self.extr_targ = copy.deepcopy(self.extr) if self.extr_targ else None
        self.rl.load(model_dir, step)
        self._load(model_dir, step)

    @abc.abstractmethod
    def _load(self, model_dir, step):
        raise NotImplementedError

    def print_log(self, logger, test_env, epoch, step, ar, test, start_time, epoch_fps):
        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('TotalEnvInteracts', (step + 1) * ar)
        logger.log_tabular('Step', step + 1)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpNum', average_only=True)
        logger.log_tabular('EpLen', average_only=True)
        if test:
            if isinstance(test_env, list):
                for i in range(len(test_env)):
                    logger.log_tabular(f'TestEpRet{i}', with_min_and_max=True)
            else:
                logger.log_tabular('TestEpRet0', with_min_and_max=True)
        logger.log_tabular('Qvals', average_only=True)
        logger.log_tabular('Qmaxs', average_only=True)
        logger.log_tabular('TQvals', average_only=True)
        logger.log_tabular('TQmaxs', average_only=True)
        logger.log_tabular('LossQ', average_only=True)
        logger.log_tabular('LossPi', average_only=True)
        
        self.rl._print_log(logger)
        self._print_log(logger)

        logger.log_tabular('Time', (time.time() - start_time)/3600)
        logger.log_tabular('FPS', epoch_fps)
        logger.dump_tabular()

    @abc.abstractmethod
    def _print_log(logger):
        raise NotImplementedError
