import copy
import torch
import torch.nn.functional as F
import numpy as np

from .algo_base import ALGOBase
from module.rl_module import MLPActor, EnsembleCritic
from common.utils import update_params, soft_update_params


class TD3(ALGOBase):
    
    def __init__(self, action_shape, action_limit, act_noise=0.1, device='cpu',
                 critic_tau=0.05, num_q=2, num_targ_q=2, hidden_dim=1024, l=2,
                 extr_latent_dim=50, std_clip=0.3, repr_dict=dict(), **kwargs):
        super().__init__(action_limit, num_q, num_targ_q)
        # Setting hyperparameters
        extr_has_fc, actor_repr_dim, critic_repr_dim = repr_dict
        actor_repr_dim = None if extr_has_fc else actor_repr_dim
        critic_repr_dim = None if extr_has_fc else critic_repr_dim
        self.critic_tau = critic_tau
        self.std_clip = std_clip
        
        # Setting modules
        self.actor = MLPActor(action_shape, hidden_dim, actor_repr_dim, extr_latent_dim,
                              l, self.action_limit, act_noise).to(device)
        self.critic = EnsembleCritic(action_shape, hidden_dim, critic_repr_dim,
                                     extr_latent_dim, l, num_q=num_q).to(device)
        self.critic_targ = EnsembleCritic(action_shape, hidden_dim, critic_repr_dim,
                                          extr_latent_dim, l, num_q=num_q).to(device)
        self.critic_targ.load_state_dict(self.critic.state_dict())

    def select_action(self, s, deterministic=False, std=None, tanh=True):
        pi = self.actor.act(s, deterministic, std, None, tanh)
        return pi.cpu().data.numpy().flatten()

    @torch.no_grad()
    def get_q_target(self, s2, r, gamma, nd, std=None):
        a2 = self.actor(s2, act_noise=std, clip=self.std_clip, with_logprob=False)
        q_pi_targ = self.critic_targ(s2, a2, False) # (num_q, batch_size)
        if self.num_targ_q < self.num_q:
            idxs = np.random.choice(self.num_q, self.num_targ_q, replace=False)
            min_q_pi_targ = q_pi_targ[idxs].min(dim=0)[0]
            max_q_pi_targ = q_pi_targ[idxs].max(dim=0)[0]
        else:
            min_q_pi_targ = q_pi_targ.min(dim=0)[0]
            max_q_pi_targ = q_pi_targ.max(dim=0)[0]
        q_targ = r + gamma * nd * min_q_pi_targ
        q_targ_max = r + gamma * nd * max_q_pi_targ
        return q_targ, q_targ_max # (batch_size,)

    def update_critic(self, s, a, r, s2, nd=None, std=None, gamma=0.99):
        q = self.critic(s, a, False) # (num_q, batch_size)
        q_targ, q_targ_max = self.get_q_target(s2, r, gamma, nd, std)
        
        loss_q = F.mse_loss(q, q_targ) * q.size(0)
        return loss_q, dict(Qvals=q.min(dim=0)[0].mean().item(), Qmaxs=q.max(dim=0)[0].mean().item(),
                            TQvals=q_targ.mean().item(), TQmaxs=q_targ_max.mean().item(), LossQ=loss_q.item()
                            ), dict(s=self.critic.forward_trunk(s).detach())

    def update_actor(self, s, std):
        a, log_pi, entropy = self.actor(s, act_noise=std, clip=self.std_clip)
        loss_pi = -self.critic(s, a).mean()
        # entropy = a.shape[1] * (0.5 * (1.0 + np.log(2 * np.pi)) + np.log(std))
        return loss_pi, dict(LossPi=loss_pi.item(), HPi=log_pi.mean().item(), Entro=entropy.mean().item(), STD=std)

    def soft_update_params(self):
        soft_update_params(self.critic, self.critic_targ, self.critic_tau)

    def _save(self, model_dir, step):
        pass

    def _load(self, model_dir, step):
        pass

    def _print_log(self, logger):
        logger.log_tabular('HPi', average_only=True)
        logger.log_tabular('Entro', average_only=True)
        logger.log_tabular('STD', average_only=True)
