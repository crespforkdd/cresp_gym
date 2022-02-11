import torch
import numpy as np
import os
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path

from common import utils
from common.buffer_trajectory import BReplayBuffer
from common.buffer import ReplayBuffer
from common.logger_tb import Logger
from common.logx import EpochLogger, setup_logger_kwargs
from common.video import VideoRecorder

from argument import parse_args
from module.make_module import init_agent
from train import train_agent

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


def main(args, device, work_dir, config):
    if args.seed == -1: 
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)
    _, domain_name, task_name = args.env.split('.')

    # make directory
    dy = '' if args.dynamic else '-ND'
    bf_ns = config['buffer_params']['nstep_rew']
    dis = '-' if bf_ns == 1 else '-g' + ('%s' % int(args.rs_discount * 100)) + '-'
    mode = f"{args.agent}{args.cf_config}-T{bf_ns}{dis}"
    if args.background:
        env_mode = '-Ba'
    elif args.camera:
        env_mode = '-Ca'
    elif args.color:
        env_mode = '-Co'
    else:
        env_mode = '-Clean'
    env_mode += f'{args.num_sources}'
    exp_name = f"{args.env}{env_mode}{dy}-b{args.batch_size}" + mode
    config['exp_name'] = exp_name

    work_dir = work_dir / 'data' / f'{args.env}'
    logger_kwargs = setup_logger_kwargs(exp_name, args.seed, work_dir)
    # Setting logger and save hyperparameters
    L = EpochLogger(**logger_kwargs)
    L.save_config(locals())
    work_dir = work_dir / f'{exp_name}' / f'{exp_name}_s{args.seed}'
    # utils.make_dir(work_dir)
    TBL = Logger(work_dir, action_repeat=args.action_repeat, use_tb=args.save_tb)
    video_dir = utils.make_dir(work_dir / 'video')
    model_dir = utils.make_dir(work_dir / 'model')
    buffer_dir = utils.make_dir(work_dir / 'buffer')

    video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
    # Setting Environment
    train_envs, test_env, obs_dict = utils.set_dcs_multisources(domain_name,
                                                                task_name,
                                                                config['buffer_params']['image_size'],
                                                                config['train_params']['action_repeat'],
                                                                dynamic=args.dynamic,
                                                                background=args.background,
                                                                camera=args.camera,
                                                                color=args.color,
                                                                test_background=args.test_background,
                                                                test_camera=args.test_camera,
                                                                test_color=args.test_color,
                                                                **config['setting'])
    obs_shape, pre_aug_obs_shape = obs_dict
    action_shape = train_envs[0].action_space.shape
    action_limit = train_envs[0].action_space.high[0]

    replay_buffer = BReplayBuffer(obs_shape=pre_aug_obs_shape,
                                  action_shape=action_shape,
                                  buffer_dir=buffer_dir,
                                  batch_size=args.batch_size,
                                  device=device,
                                  **config['buffer_params'])

    config.update(dict(obs_shape=obs_shape, batch_size=args.batch_size, device=device))
    config['algo_params'].update(dict(action_shape=action_shape,
                                      action_limit=action_limit,
                                      device=device))

    agent = init_agent(args.agent, config)

    train_agent(train_envs=train_envs,
                test_env=test_env,
                agent=agent,
                replay_buffer=replay_buffer,
                L=L,
                TBL=TBL,
                video=video,
                model_dir=model_dir,
                num_updates=args.num_updates,
                device=device,
                **config['train_params'])

    for env in train_envs:
        env.close()
    test_env.close()


if __name__ == '__main__':

    args = parse_args()
    for i in args.seed_list:
        cuda_id = f"cuda:{args.cuda_id}"
        device = torch.device(cuda_id if args.cuda else "cpu")
        work_dir = Path.cwd()
        config = utils.read_config(args, work_dir / args.config_dir)
        torch.multiprocessing.set_start_method('spawn', force=True)
        args.seed, config['setting']['seed'] = i, i
        main(args, device, work_dir, config)
