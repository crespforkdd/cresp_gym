import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from common.utils import gaussian_logprob, squash, EnsembleLinear, weight_init, mlp


class SGMLPActor(nn.Module):
    
    def __init__(
        self, action_shape, hidden_dim, repr_dim, encoder_feature_dim,
        log_std_min, log_std_max, l, action_limit, r_dim=1, k_dim=1, nstep_rew=1
    ):
        super(SGMLPActor, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.act_limit = action_limit
        self.state_dim = encoder_feature_dim
        self.repr_dim = repr_dim
        self.r_dim = r_dim
        self.k_dim = k_dim
        self.nstep_rew = nstep_rew
        self.l = l
        self.cf_input_dim = self.state_dim + action_shape[0] * nstep_rew + nstep_rew
        self.cf_output_dim = r_dim * 2

        if repr_dim is not None:
            self.trunk = mlp(repr_dim, 0, self.state_dim, 0, nn.LayerNorm,
                            handle_dim=self.state_dim)
        self.fc = mlp(self.state_dim, hidden_dim, hidden_dim, l-1)
        self.pi = mlp(hidden_dim, 0, 2 * action_shape[0], 0)
        self.cfunc = mlp(self.cf_input_dim, hidden_dim, self.cf_output_dim, l)
        self.t_mu = nn.Parameter(torch.zeros(nstep_rew))
        self.t_log_std = nn.Parameter(torch.ones_like(self.t_mu)*2.0/3.0) # Normal
        self.infos = dict()
        self.apply(weight_init)

    def t(self, batch_size, nsr=None, dim=None):
        if nsr is None:
            nsr = self.nstep_rew
        log_std = torch.tanh(self.t_log_std)
        log_std = 3 * log_std - 2.0
        self.infos['t_mu'] = self.t_mu
        self.infos['t_std'] = log_std.exp()
        std = log_std.exp()
        if dim is None:
            noise = torch.randn(self.k_dim, batch_size, nsr).to(self.t_mu.device)
        else:
            noise = torch.randn(self.k_dim, batch_size, nsr, dim).to(self.t_mu.device)
        t = self.t_mu + noise * std
        self.infos['t'] = t
        return t # (k_dim, batch_size, nstep_rew)

    def pr(self, state, action, t=None):
        state = self.forward_trunk(state)
        if action.ndim == 2:
            cat_sa = torch.cat([state, action], dim=-1)
            p_next_z_r = self.rew_fc(cat_sa)
            return p_next_z_r
        assert t.ndim == 3, print('t', t.size()) # (k_dim, b, nstep_rew)
        cat_sa = torch.cat([state] + list(action), dim=-1) # (b, s+a*nsr)
        cat_sa = cat_sa.unsqueeze(0).expand(t.size(0), *cat_sa.size())
        cat_sat = torch.cat([cat_sa, t], dim=-1) # (k_dim, b, s+a/a*nsr+nstep_rew)
        out_cos, out_sin = self.cfunc(cat_sat).chunk(2, -1) # (k_dim, b, er_dim)
        out = torch.cat([out_cos, out_sin], dim=0) # (2*k_dim, b, er_dim)
        return out.transpose(0, -1) # (er_dim, b, 2*k)

    def forward_trunk(self, obs, tanh=True):
        if self.repr_dim is None:
            return obs
        state = self.trunk(obs)
        self.infos['ln'] = state
        if tanh:
            state = torch.tanh(state)
            self.infos['tanh'] = state
        return state

    def output(self, pi):
        if pi is None:
            return None
        return self.act_limit * pi

    def gaussian(self, state, tanh=True, tanh_mu=False):
        state = self.fc(self.forward_trunk(state, tanh))
        mu, log_std = self.pi(state).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        self.infos['mu'] = mu
        self.infos['std'] = log_std.exp()
        return mu, log_std, state

    def forward(self, state, compute_pi=True, with_logprob=True, tanh=True, output=False):
        mu, log_std, _ = self.gaussian(state, tanh)

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None

        if with_logprob:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        if output:
            return self.output(mu), self.output(pi), log_pi, log_std, state
        return self.output(mu), self.output(pi), log_pi, log_std

    def act(self, state, deterministic=False, tanh=True):
        mu_action, pi_action, log_pi, _ = self.forward(state, not deterministic, not deterministic, tanh)
        if deterministic:
            return mu_action
        return pi_action, log_pi

    @torch.no_grad()
    def log_prob_acts(self, state, action, tanh=True, trunk=False):
        state = self.forward_trunk(state, tanh) if not trunk else state
        mu, log_std = self.pi(self.fc(state)).chunk(2, dim=-1)
        if action.ndim == 3:
            mu = mu.unsqueeze(1).expand(*action.size())
            log_std = log_std.unsqueeze(1).expand(*action.size())

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        if (action >= 1.0).any() or (action <= -1.0).any():
            action[action >= 1.0] = 1.0 - 1e-6
            action[action <= -1.0] = -1.0 + 1e-6
        org_action = torch.atanh(action)
        noise = (org_action - mu) / log_std.exp()

        mu = torch.tanh(mu)
        log_pi = gaussian_logprob(noise, log_std)
        log_pi -= torch.log(F.relu(1 - action.pow(2)) + 1e-6).sum(-1, keepdim=True)
        log_pi = log_pi.squeeze(-1)
        return log_pi

    def log(self, L, step, log_freq):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        if self.repr_dim is not None:
            L.log_param('train_actor/fc', self.trunk[0], step)
            L.log_param('train_actor/ln', self.trunk[1], step)
        for i in range(self.l):
            L.log_param('train_actor/pi_fc%d' % i, self.fc[i * 2], step)
        L.log_param('train_actor/pi', self.pi[0], step)


from common.utils import TruncatedNormal

class MLPActor(nn.Module):

    def __init__(self, action_shape, hidden_dim, repr_dim, encoder_feature_dim,
                 l, act_limit, act_noise=0.1, eps=1e-6):
        super(MLPActor, self).__init__()
        self.act_limit = act_limit
        self.act_noise = act_noise
        self.state_dim = encoder_feature_dim
        self.repr_dim = repr_dim
        self.eps = eps

        if repr_dim is not None:
            self.trunk = mlp(repr_dim, 0, self.state_dim, 0, nn.LayerNorm,
                            handle_dim=self.state_dim)
        self.pi = mlp(self.state_dim, hidden_dim, action_shape[0], l, nn.Tanh)
        self.infos = dict()
        self.apply(weight_init)

    def forward_trunk(self, obs, tanh=True):
        if self.repr_dim is None:
            return obs
        state = self.trunk(obs)
        self.infos['ln'] = state
        if tanh:
            state = torch.tanh(state)
            self.infos['tanh'] = state
        return state

    def forward(self, obs, deterministic=False, act_noise=None, clip=None, tanh=True, with_logprob=True):
        state = self.forward_trunk(obs, tanh)
        mu = self.act_limit * self.pi(state)
        self.infos['mu'] = mu

        if act_noise is None:
            act_noise = self.act_noise
        dist = TruncatedNormal(mu, torch.ones_like(mu) * act_noise)

        if deterministic:
            pi_action = dist.mean
        else:
            pi_action = dist.sample(clip=clip)
        
        if with_logprob:
            log_pi = dist.log_prob(pi_action).sum(-1, keepdim=True)
            return pi_action, log_pi, dist.entropy().sum(dim=-1)

        return pi_action

    def act(self, state, deterministic=False, act_noise=None, clip=None, tanh=True):
        return self.forward(state, deterministic, act_noise, clip, tanh, False)

    def log(self, L, step, log_freq):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        if self.repr_dim is not None:
            L.log_param('train_actor/fc', self.trunk[0], step)
            L.log_param('train_actor/ln', self.trunk[1], step)
        for i in range(3):
            L.log_param('train_actor/pi_fc%d' % i, self.pi[i * 2], step)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(self, action_shape, hidden_dim, repr_dim, encoder_feature_dim, l=2,
                 output_mod=None, num_q=2, output_dim=1):
        super(Critic, self).__init__()

        self.state_dim = encoder_feature_dim
        self.output_dim = output_dim
        self.repr_dim = repr_dim
        self.num_q = num_q
        
        if repr_dim is not None:
            self.trunk = nn.Sequential(nn.Linear(repr_dim, encoder_feature_dim),
                                       nn.LayerNorm(encoder_feature_dim))
        # self.q1 = mlp(self.state_dim + action_shape[0], hidden_dim, output_dim, l, output_mod)
        self.q1 = nn.Sequential(nn.Linear(encoder_feature_dim+action_shape[0], hidden_dim),
                                nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(nn.Linear(encoder_feature_dim+action_shape[0], hidden_dim),
                                nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
                                nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))
        self.infos = dict()
        self.apply(weight_init)

    def forward_trunk(self, obs, tanh=True):
        if self.repr_dim is None:
            return obs
        state = self.trunk(obs)
        self.infos['ln'] = state
        if tanh:
            state = torch.tanh(state)
            self.infos['tanh'] = state
        return state

    def forward(self, state, action, tanh=True):
        state = self.forward_trunk(state)
        assert state.size()[:-1] == action.size()[:-1], print(state.size(), action.size())
        sa = torch.cat([state, action], 1)
        q1 = torch.squeeze(self.q1(sa), -1) # q:(batch_size,)
        q2 = torch.squeeze(self.q2(sa), -1)
        self.infos['q1'] = q1
        self.infos['q2'] = q2
        return q1, q2

    def forward_q(self, state, action):
        assert state.size()[:-1] == action.size()[:-1], print(state.size(), action.size())
        sa = torch.cat([state, action], 1)
        q1 = torch.squeeze(self.q1(sa), -1) # q:(batch_size,)
        q2 = torch.squeeze(self.q2(sa), -1)
        self.infos['q1'] = q1
        self.infos['q2'] = q2
        return q1, q2

    def Q1(self, state, action, tanh=True):
        state = self.forward_trunk(state)
        assert state.size()[:-1] == action.size()[:-1], print(state.size(), action.size())
        sa = torch.cat([state, action], 1)
        return torch.squeeze(self.q1(sa), -1)

    def log(self, L, step, log_freq):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        if self.repr_dim is not None:
            L.log_param('train_critic/fc', self.trunk[0], step)
            L.log_param('train_critic/ln', self.trunk[1], step)
        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.q1[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.q2[i * 2], step)


class EnsembleCritic(nn.Module):

    def __init__(self, action_shape, hidden_dim, repr_dim, encoder_feature_dim, l=2,
                 output_mod=None, num_q=2, output_dim=1, handle_dim=None):
        super(EnsembleCritic, self).__init__()

        self.state_dim = encoder_feature_dim
        self.output_dim = output_dim
        self.num_q = num_q
        self.repr_dim = repr_dim
        
        if repr_dim is not None:
            self.trunk = mlp(repr_dim, 0, self.state_dim, 0, nn.LayerNorm, handle_dim=self.state_dim)
        self.q = mlp(
            self.state_dim + action_shape[0], hidden_dim, output_dim, l,
            output_mod, handle_dim=handle_dim, channel_dim=num_q, linear=EnsembleLinear
        )
        self.infos = dict()
        self.apply(weight_init)

    def forward_trunk(self, obs, tanh=True):
        if self.repr_dim is None:
            return obs
        state = self.trunk(obs)
        self.infos['ln'] = state
        if tanh:
            state = torch.tanh(state)
            self.infos['tanh'] = state
        return state

    def forward(self, state, action, minimize=True, tanh=True):
        state = self.forward_trunk(state, tanh)
        assert state.size()[:-1] == action.size()[:-1], print(state.size(), action.size())
        sa = torch.cat([state, action], -1)
        q = self.q(sa)

        for i in range(q.size(0)):
            self.infos['q%s' % (i + 1)] = q[i]

        if minimize:
            q = q.min(dim=0)[0] if q.size(0) == self.num_q else q # q:(batch_size,)
            self.infos['q_min'] = q
        return q

    def forward_q(self, state, action, minimize=True):
        assert state.size()[:-1] == action.size()[:-1], print(state.size(), action.size())
        sa = torch.cat([state, action], -1)
        q = self.q(sa) # (batch_size, 1) or (num_q, batch_size, 1)
        q = torch.squeeze(q, -1) if q.size(-1) == 1 else q # q:(num_q, batch_size)

        for i in range(q.size(0)):
            self.infos['q%s' % (i + 1)] = q[i]

        if minimize:
            q = q.min(dim=0)[0] if q.size(0) == self.num_q else q # q:(batch_size,)
            self.infos['q_min'] = q
        return q

    def Q1(self, state, action=None, tanh=True):
        state = self.forward_trunk(state, tanh)
        if action is None:
            sa = state
        else:
            assert state.size()[:-1] == action.size()[:-1], print(state.size(), action.size())
            sa = torch.cat([state, action], -1)
        q = self.q(sa)
        q = torch.squeeze(q, -1) if q.size(-1) == 1 else q
        q = q[0] if q.size(0) == self.num_q else q
        return q

    def log(self, L, step, log_freq):
        if not log_freq or step % log_freq != 0:
            return

        for k, v in self.infos.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        if self.repr_dim is not None:
            L.log_param('train_critic/fc', self.trunk[0], step)
            L.log_param('train_critic/ln', self.trunk[1], step)
        for i in range(3):
            L.log_param('train_critic/q_ensemble_fc%d' % i, self.q[i * 2], step)
