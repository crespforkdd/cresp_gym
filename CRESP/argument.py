import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--env', type=str, default='dmc.hopper.hop')
    parser.add_argument('--action_repeat', '-ar', default=2, type=int)
    # algorithm
    parser.add_argument('--base', type=str, default='sac', choices=['sac', 'td3'])
    parser.add_argument('--agent', type=str, default='cresp')
    # aug
    parser.add_argument('--data_aug', default='shift', type=str)
    # training settings
    parser.add_argument('--num_sources', default=2, type=int)
    parser.add_argument('--dynamic', '-d', default=False, action='store_true')
    parser.add_argument('--background', '-bg', default=False, action='store_true')
    parser.add_argument('--camera', '-ca', default=False, action='store_true')
    parser.add_argument('--color', '-co', default=False, action='store_true')
    parser.add_argument('--test_background', '-tbg', default=False, action='store_true')
    parser.add_argument('--test_camera', '-tca', default=False, action='store_true')
    parser.add_argument('--test_color', '-tco', default=False, action='store_true')
    # training hypers
    parser.add_argument('--batch_size', '-bs', default=256, type=int)
    parser.add_argument('--nstep_rew', '-nsr', default=1, type=int)
    parser.add_argument('--num_updates', default=1, type=int)
    parser.add_argument('--extr_q_update_freq', '-euf', default=10, type=int)
    parser.add_argument('--actor_mode', default='sum', type=str)
    # module
    parser.add_argument('--no-default', default=False, action='store_true')
    
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--alpha_lr', default=5e-4, type=float)
    parser.add_argument('--extr_lr', '-elr', default=5e-4, type=float)
    parser.add_argument('--rs_discount', '-rdis', default=0.8, type=float)
    parser.add_argument('--cf-config', '-cfc', default='r1-k256', type=str)
    # save
    parser.add_argument('--config_dir', type=str, default='config')
    parser.add_argument('--save_tb', default=True, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=True, action='store_true')
    # seed
    parser.add_argument('--seed_list', '-s', nargs='+', type=int,
                         default=[0, 1, 2, 3, 4])
    parser.add_argument('--seed', default=1, type=int)
    # cuda
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--cuda_id', type=int, default=0)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args