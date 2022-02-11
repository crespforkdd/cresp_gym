import numpy as np
import copy
import torch
import torch.nn as nn

import common.utils as utils
from module.extr_module import make_extr


def init_extractor(obs_shape, device, extr_config):
    extr = make_extr(obs_shape=obs_shape, **extr_config).to(device)
    extr_targ = None
    if extr_config['targ_extr']:
        extr_targ = copy.deepcopy(extr)
    extr_tau = extr_config['extr_tau']
    out_dict = dict(extr=extr, extr_targ=extr_targ, extr_tau=extr_tau)
    return out_dict


from algo import *

_AVAILABLE_ALGORITHM = {'sac': SAC, 'td3': TD3}

def init_algo(algo, repr_dict, algo_config, critic_type='common'):
    assert algo in _AVAILABLE_ALGORITHM
    return _AVAILABLE_ALGORITHM[algo](repr_dict=repr_dict, critic_type=critic_type, **algo_config)


from agent import *
from augmentation import *

_AVAILABLE_AGENT = {'cresp': CRESPAgent}

_AVAILABLE_AUG_FUNC = {
    'shift': RandomShiftsAug,
}


def init_aug_func(afunc1, image_pad=None):
    assert 'combo' in afunc1 or afunc1 in _AVAILABLE_AUG_FUNC
    aug_shift = _AVAILABLE_AUG_FUNC['shift'](image_pad) if image_pad is not None \
            else _AVAILABLE_AUG_FUNC['shift']()

    aug_funcs = aug_shift if afunc1 == 'shift' else _AVAILABLE_AUG_FUNC[afunc1]()
    return aug_funcs


def init_agent(agt, config):
    assert agt in _AVAILABLE_AGENT
    if config['evaluation']:
        aug_funcs = None
    else:
        aug_funcs = init_aug_func(config['data_aug'], config['agent_params']['image_pad'])
    print(aug_funcs)
    return _AVAILABLE_AGENT[agt](aug_funcs, config=config, **config['agent_params'])
