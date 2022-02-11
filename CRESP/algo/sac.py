import copy
import torch
import torch.nn.functional as F
import numpy as np

from .algo_base import ALGOBase
from module.rl_module import SGMLPActor, EnsembleCritic
from common.utils import update_params, soft_update_params

_AVAILABLE_CRITIC = {'common': EnsembleCritic}


class SAC(ALGOBase):

    def __init__(self, action_shape, action_limit, device='cpu', critic_tau=0.05,
                 num_q=2, num_targ_q=2, hidden_dim=1024, l=2, extr_latent_dim=50,
                 repr_dict=dict(), init_temperature=0.1, actor_log_std_min=-10,
                 actor_log_std_max=2, critic_type='common', cfun=False, **kwargs):
        super().__init__(action_limit, num_q, num_targ_q)

        extr_has_fc, actor_repr_dim, critic_repr_dim = repr_dict
        # self.is_fc = not extr_has_fc
        actor_repr_dim = None if extr_has_fc else actor_repr_dim
        critic_repr_dim = None if extr_has_fc else critic_repr_dim
        self.critic_tau = critic_tau
        self.extr_latent_dim = extr_latent_dim
        if isinstance(l, list):
            self.l, l_q, l_pi = l[0], l[0], l[1]
        else:
            self.l, l_q, l_pi = l, l, l
        r_dim = kwargs['r_dim'] if 'r_dim' in kwargs.keys() else 1
        k_dim = kwargs['k_dim'] if 'k_dim' in kwargs.keys() else 1
        nstep_rew = kwargs['nstep_rew'] if 'nstep_rew' in kwargs.keys() else 1

        # Setting modules
        self.actor = SGMLPActor(
            action_shape, hidden_dim, actor_repr_dim, extr_latent_dim,
            actor_log_std_min, actor_log_std_max, l_pi, self.action_limit,
            r_dim=r_dim, k_dim=k_dim, nstep_rew=nstep_rew
        ).to(device)
        self.critic = _AVAILABLE_CRITIC[critic_type](
            action_shape, hidden_dim, critic_repr_dim, extr_latent_dim,
            l_q, num_q=num_q
        ).to(device)
        self.critic_targ = copy.deepcopy(self.critic)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, s, deterministic=False, tanh=False):
        out = self.actor.act(s, deterministic, tanh)
        if deterministic:
            return out.cpu().data.numpy().flatten()
        return out[0].cpu().data.numpy().flatten(), out[1].item()

    @torch.no_grad()
    def get_q_target(self, s2, r, gamma, nd):
        _, a2, logp_a2, _ = self.actor(s2)
        q_pi_targ = self.critic_targ(s2, a2, False).min(-1)[0] # (num_q, batch_size)
        if self.num_targ_q < self.num_q:
            idxs = np.random.choice(self.num_q, self.num_targ_q, replace=False)
            min_q_pi_targ = q_pi_targ[idxs].min(dim=0)[0] - self.alpha * logp_a2
            max_q_pi_targ = q_pi_targ[idxs].max(dim=0)[0] - self.alpha * logp_a2
        else:
            min_q_pi_targ = q_pi_targ.min(dim=0)[0] - self.alpha * logp_a2
            max_q_pi_targ = q_pi_targ.max(dim=0)[0] - self.alpha * logp_a2
        q_targ = r + gamma * nd * min_q_pi_targ
        q_targ_max = r + gamma * nd * max_q_pi_targ
        return q_targ, q_targ_max # (batch_size,)

    @torch.no_grad()
    def get_q1_target(self, idx, s2, r, gamma, nd):
        _, a2, logp_a2, _ = self.actor(s2)
        q_pi_targ = self.critic_targ(s2, a2, False)[idx].min(-1)[0] # (num_q, batch_size)
        q_targ = r + gamma * nd * q_pi_targ # (batch_size,)
        return q_targ

    def calculate_q(self, s, a, num, batch_size):
        _s = self.critic.forward_trunk(s)
        _s = _s.view(num, 1, batch_size, -1) if _s.ndim == 2 else _s # (num, 1, batch_size, enc_dim)
        if a.size(0) == batch_size:
            _a = a.expand((num, *a.size())).unsqueeze(1) # (num, 1, batch_size, act_dim)
        else:
            _a = a.view(num, 1, batch_size, -1)
        return self.critic.forward_q(_s, _a, False)  # (num, num_q, batch_size)

    def update_critic(self, s, a, r=None, s2=None, nd=None, num=1, gamma=0.99, q_targ=None):
        q_info_dict = dict()
        with torch.no_grad():
            if q_targ is None:
                q_targ, q_targ_max = self.get_q_target(s2, r, gamma, nd)
                if num is not None and isinstance(num, int):
                    q_targ = q_targ.view(num, -1).mean(0)
                q_info_dict = dict(TQvals=q_targ, TQmaxs=q_targ_max)
            num = 1 if a.size(0) == s.size(0) else num
            batch_size = a.size(0)

        q = self.calculate_q(s, a, num, batch_size) # (num, num_q, batch_size)
        loss_q = F.mse_loss(q, q_targ.view(-1, 1, q_targ.size(-1)).expand(*q.size()), reduction='none')

        q_info_dict.update(dict(Qvals=q.min(dim=1)[0].mean([0, 1]), Qmaxs=q.max(dim=1)[0].mean([0, -1]), LossQ=loss_q.mean()))
        return loss_q.sum([0, 1]).mean(), q_info_dict

    def select_q_pi(self, q_pi, mode):
        if mode == 'min':
            q_pi = torch.min(q_pi, dim=0)[0]
        elif mode == 'mean':
            q_pi = q_pi.mean(0)
        elif mode == 'sample':
            idx = np.random.choice(n, 2, replace=False)
            q_pi = q_pi[idx].mean(0)
        elif mode == 'sum':
            q_pi = q_pi.sum(0)
        elif mode == 'single':
            q_pi = q_pi[0]
        return q_pi # q_pi: (batch_size,)

    def update_actor(self, s, mode='min', num=1):
        _, a, logp_a, log_std = self.actor(s)
        q = self.calculate_q(s, a, num, s.size(0)//num) # (num, num_q, batch_size)
        q = torch.min(q, dim=1)[0] - self.alpha.detach() * logp_a.view(num, -1) # (num, batch_size)
        loss_pi = - self.select_q_pi(q, mode)
        loss_alpha = (self.alpha * (-logp_a - self.target_entropy).detach()).mean()

        pi_info_dict = dict(HPi=-logp_a.view(num, -1).mean(0), LossPi=loss_pi, Alpha=self.alpha, LossAlpha=loss_alpha)
        return loss_pi.mean(), loss_alpha, pi_info_dict

    def soft_update_params(self):
        soft_update_params(self.critic, self.critic_targ, self.critic_tau)

    def _save(self, model_dir, step):
        pass

    def _load(self, model_dir, step):
        pass

    def _print_log(self, logger):
        logger.log_tabular('HPi', average_only=True)
        logger.log_tabular('Alpha', average_only=True)
        logger.log_tabular('LossAlpha', average_only=True)
