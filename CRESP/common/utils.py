import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import gym
import yaml
import random
import numpy as np
from collections import deque
from torch.distributions import Normal
from torch.distributions.utils import _standard_normal
import dmc2gym
from distracting_control import suite_utils
from common import suite, wrappers, logx


_ACTION_REPEAT = {
    'dmc.ball_in_cup.catch': 4, 'dmc.cartpole.swingup': 8, 'dmc.cheetah.run': 4,
    'dmc.finger.spin': 2, 'dmc.reacher.easy': 4, 'dmc.walker.walk': 2
}

_DIFFICULTY = ['easy', 'medium', 'hard']
_DISTRACT_MODE = ['train', 'training', 'val', 'validation']


def read_config(args, config_dir):
    # read common.yaml
    with open(config_dir / 'common.yaml') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    # read algo.yaml and update config.algo_params
    with open(config_dir / 'algo.yaml') as f:
        alg_config = yaml.load(f, Loader=yaml.SafeLoader)
    config['algo_params'].update(alg_config)

    # read agent.yaml and update config
    with open(config_dir / 'agent.yaml') as f:
        agent_config = yaml.load(f, Loader=yaml.SafeLoader)

    for key in agent_config[args.agent].keys():
        if isinstance(agent_config[args.agent][key], dict):
            config[key].update(agent_config[args.agent][key])
        else:
            config[key] = agent_config[args.agent][key]

    config['train_params']['action_repeat'] = _ACTION_REPEAT[args.env]
    config['agent_base_params']['num_sources'] = config['setting']['num_sources']
    # update config via args
    if args.no_default:
        kwargs = {
            'data_aug': args.data_aug,
            'buffer_params': {
                'nstep_rew': args.nstep_rew,
                'save_buffer': args.save_buffer,
            },
            'agent_params': {
                'extr_lr': args.extr_lr,
                'critic_lr': args.critic_lr,
                'actor_lr': args.actor_lr,
                'alpha_lr': args.alpha_lr,
                'actor_mode': args.actor_mode,
                'discount': args.rs_discount,
                'sm_mode': args.cf_config,
                'extr_q_update_freq': args.extr_q_update_freq,
                'rs_discount': args.rs_discount,
            },
            'algo_params': {
                'nstep_rew': args.nstep_rew
            },
            'train_params': {
                'action_repeat': args.action_repeat,
            },
            'setting': {
                'num_sources': args.num_sources,
                'dynamic': args.dynamic,
                'background': args.background,
                'camera': args.camera,
                'color': args.color,
                'num_videos': args.num_videos
            }
        }
        for key in kwargs.keys():
            if isinstance(kwargs[key], dict):
                config[key].update(kwargs[key])
            else:
                config[key] = kwargs[key]
        config['agent_base_params']['num_sources'] = args.num_sources

    config['agent_base_params']['action_repeat'] = config['train_params']['action_repeat']
    config['train_params']['eval_freq'] = config['steps_per_epoch'] // config['train_params']['action_repeat']
    config['train_params']['total_steps'] = config['total_steps'] // config['train_params']['action_repeat'] + 1
    return config


def make_env(domain_name,
             task_name,
             seed=1,
             image_size=84,
             action_repeat=2,
             frame_stack=3,
             background_dataset_path=None,
             difficulty=None,
             background_kwargs=None,
             camera_kwargs=None,
             color_kwargs=None):

    env = suite.load(domain_name,
                     task_name,
                     difficulty,
                     seed=seed,
                     background_dataset_path=background_dataset_path,
                     background_kwargs=background_kwargs,
                     camera_kwargs=camera_kwargs,
                     color_kwargs=color_kwargs,
                     visualize_reward=False)

    camera_id = 2 if domain_name == "quadruped" else 0
    env = wrappers.DMCWrapper(env,
                              seed=seed,
                              from_pixels=True,
                              height=image_size,
                              width=image_size,
                              camera_id=camera_id,
                              frame_skip=action_repeat)
    env = wrappers.FrameStack(env, k=frame_stack)
    return env


def set_dcs_multisources(domain_name, task_name, image_size, action_repeat,
                         seed, frame_stack, background_dataset_path, num_sources,
                         difficulty='hard', dynamic=True, distract_mode='train',
                         background=False, camera=False, color=False, num_videos=1,
                         test_background=False, test_camera=False, test_color=False,
                         video_start_idxs=None, scale=None, test_scale=0.3,
                         color_scale=None, test_color_scale=None, **kargs):
    assert difficulty in _DIFFICULTY
    assert distract_mode in _DISTRACT_MODE
    if video_start_idxs is None:
        video_start_idxs = [num_videos * i for i in range(num_sources)]

    train_envs = []
    for i in range(num_sources):
        background_kwargs, camera_kwargs, color_kwargs = None, None, None

        if background:
            if num_videos is None:
                num_videos = suite_utils.DIFFICULTY_NUM_VIDEOS[difficulty]
            background_kwargs = suite_utils.get_background_kwargs(
                domain_name, num_videos, dynamic, background_dataset_path, distract_mode)
            background_kwargs['start_idx'] = video_start_idxs[i]
        
        if camera:
            if scale is None:
                scale = suite_utils.DIFFICULTY_SCALE[difficulty]
            scale += i * scale
            camera_kwargs = suite_utils.get_camera_kwargs(domain_name, scale, dynamic)
        
        if color:
            if color_scale is None:
                color_scale = suite_utils.DIFFICULTY_SCALE[difficulty]
            color_scale += i * color_scale
            color_kwargs = suite_utils.get_color_kwargs(color_scale, dynamic)

        env = make_env(domain_name, task_name, seed, image_size,
                       action_repeat, frame_stack, background_dataset_path,
                       background_kwargs=background_kwargs,
                       camera_kwargs=camera_kwargs, color_kwargs=color_kwargs)
        env.seed(seed)
        train_envs.append(env)

    # stack several consecutive frames together
    obs_shape = (3 * frame_stack, image_size, image_size)
    pre_aug_obs_shape = (3, image_size, image_size)

    # Setting Test Environment
    test_seed = np.random.randint(100, 100000) + seed
    background_kwargs, camera_kwargs, color_kwargs = None, None, None

    if test_background:
        background_kwargs = suite_utils.get_background_kwargs(
            domain_name, None, dynamic, background_dataset_path, 'val')

    if test_camera:
        if test_scale is None:
            test_scale = suite_utils.DIFFICULTY_SCALE[difficulty]
        camera_kwargs = suite_utils.get_camera_kwargs(domain_name, test_scale, dynamic)

    if test_color:
        if test_color_scale is None:
            test_color_scale = suite_utils.DIFFICULTY_SCALE[difficulty]
        color_kwargs = suite_utils.get_color_kwargs(test_color_scale, dynamic)

    test_env = make_env(domain_name, task_name, seed, image_size,
                        action_repeat, frame_stack, background_dataset_path,
                        background_kwargs=background_kwargs,
                        camera_kwargs=camera_kwargs, color_kwargs=color_kwargs)
    test_env.seed(test_seed)

    return train_envs, test_env, (obs_shape, pre_aug_obs_shape)


def update_linear_schedule(optimizer, lr):
    """Decreases the learning rate linearly"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def update_params(optim, loss, retain_graph=False, grad_cliping=False, networks=None):
    if not isinstance(optim, dict):
        optim = dict(optimizer=optim)
    for opt in optim:
        optim[opt].zero_grad()
    loss.backward(retain_graph=retain_graph)
    # Clip norms of gradients to stebilize training.
    if grad_cliping:
        try:
            for net in networks:
                nn.utils.clip_grad_norm_(net.parameters(), grad_cliping)
        except:
            nn.utils.clip_grad_norm_(networks.parameters(), grad_cliping)
    for opt in optim:
        optim[opt].step()



def cosine_similarity(z1, z2):
    z1 = z1 / torch.norm(z1, dim=-1, p=2, keepdim=True)
    z2 = z2 / torch.norm(z2, dim=-1, p=2, keepdim=True) 
    similarity = z1 @ z2.transpose(-1, -2)
    return similarity


def compute_cl_loss(z1, z2, labels=None, mask=None, temperature=1.0, output_acc=False):
    similarity = cosine_similarity(z1, z2) / temperature
    if similarity.ndim == 3:
        similarity = similarity.squeeze(1)
    if mask is not None:
        if (mask.sum(-1) != 1.0).any():
            similarity[mask] = similarity[mask] * 0.0
        else:
            similarity = similarity[~mask].view(similarity.size(0), -1)
    with torch.no_grad():
        if labels is None:
            labels = torch.arange(z1.size(0)).to(z1.device)
            target = torch.eye(z1.size(0), dtype=torch.bool).to(z1.device)
        else:
            target = F.one_hot(labels, similarity.size(1)).to(z1.device)
        pred_prob = torch.softmax(similarity, dim=-1)
        i = pred_prob.max(dim=-1)[1]
        accuracy = (i==labels).sum(-1) / labels.size(0)
        if accuracy.ndim != 1:
            accuracy = accuracy.mean()
        # accuracy = (pred_prob * target).sum(-1)
        diff = pred_prob - target.float()
    loss = (similarity * diff).sum(-1).mean(-1)
    if output_acc:
        return loss, accuracy
    return loss#, pred_prob, accuracy


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        # os.mkdir(dir_path)
        dir_path.mkdir()
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class TruncatedNormal(Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


class train_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(True)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def freeze_module(module):
    if isinstance(module, list) or isinstance(module, tuple):
        for m in module:
            for param in m.parameters():
                param.requires_grad = False
    else:
        for param in module.parameters():
            param.requires_grad = False


def activate_module(module):
    if isinstance(module, list) or isinstance(module, tuple):
        for m in module:
            for param in m.parameters():
                param.requires_grad = True
    else:
        for param in module.parameters():
            param.requires_grad = True


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
        log_pi = log_pi.squeeze(-1)
    return mu, pi, log_pi


class EnsembleLinear(nn.Module):

    def __init__(self, in_features, out_features, in_channels, bias=True):
        super(EnsembleLinear, self).__init__()
        
        self.in_channels = in_channels
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((in_channels, in_features, out_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # input: (batch_size, in_features)
        output = input @ self.weight + self.bias if self.bias is not None else input @ self.weight
        return output # output: (in_channels, batch_size, out_features)

    def extra_repr(self):
        return 'in_features={}, out_features={}, in_channels={}, bias={}'.format(
            self.in_features, self.out_features, self.in_channels, self.bias is not None
        )

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, EnsembleLinear) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def mlp(input_dim=None, hidden_dim=1024, output_dim=None, hidden_depth=2,
        output_mod=None, inplace=False, handle_dim=None, channel_dim=1, linear=nn.Linear):
    '''
    output_mod:     output activation function
        output_mod=nn.ReLU(inplace):            inplace-->False or True;
        output_mod=nn.LayerNorm(handle_dim):    handle_dim-->int
        output_mod=nn.Softmax(handle_dim):      handle_dim-->0 or 1
    linear:         choice[nn.Linear, EnsembleLinear]
        linear=EnsembleLinear:                  channel_dim-->int: ensemble number
    '''
    if isinstance(linear, nn.Linear):
        linear = lambda n_input, n_output, channel_dim: nn.Linear(n_input, n_output)
    elif linear == 'spectral_norm':
        linear = lambda n_input, n_output, channel_dim: nn.utils.spectral_norm(nn.Linear(n_input, n_output))
    elif isinstance(linear, EnsembleLinear):
        linear = lambda n_input, n_output, channel_dim: EnsembleLinear(
            in_features=n_input, out_features=n_output, in_channels=channel_dim
        )

    if hidden_depth == 0:
        mods = [linear(input_dim, output_dim, channel_dim)]
    else:
        mods = [linear(input_dim, hidden_dim, channel_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [linear(hidden_dim, hidden_dim, channel_dim), nn.ReLU(inplace=True)]
        mods.append(linear(hidden_dim, output_dim, channel_dim))
    if output_mod is not None:
        try:
            mods.append(output_mod(inplace=inplace))
        except:
            if handle_dim in [0, 1, -1]:
                mods.append(output_mod(dim=handle_dim))
            elif handle_dim is not None:
                mods.append(output_mod(handle_dim))
            else:
                mods.append(output_mod())
    trunk = nn.Sequential(*mods)
    return trunk
