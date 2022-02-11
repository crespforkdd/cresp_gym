import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path
import time

from common import utils
from common.buffer import ReplayBuffer
from common.logger_tb import Logger
from common.logx import EpochLogger, setup_logger_kwargs
from common.video import VideoRecorder
from module.make_module import init_agent
from distracting_control import suite_utils

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True
import argparse

_DIFFICULTY = ['easy', 'medium', 'hard']
_DISTRACT_MODE = ['train', 'training', 'val', 'validation']


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--env', type=str, default='dmc.cheetah.run')
    parser.add_argument('--action_repeat', '-ar', default=4, type=int)
    parser.add_argument('--image_size', default=84, type=int)
    # algorithm
    parser.add_argument('--base', type=str, default='sac', choices=['sac', 'td3'])
    parser.add_argument('--agent', type=str, default='drqaug', choices=['drq', 'drqaug', 'drqrew'])
    # aug
    parser.add_argument('--data_aug', default='shift', type=str)
    # training settings
    parser.add_argument('--num_sources', default=20, type=int)
    parser.add_argument('--dynamic', '-d', default=False, action='store_true')
    parser.add_argument('--background', '-bg', default=False, action='store_true')
    parser.add_argument('--camera', '-ca', default=False, action='store_true')
    parser.add_argument('--color', '-co', default=False, action='store_true')
    parser.add_argument('--num_videos', '-nv', default=1, type=int)
    # training hypers
    parser.add_argument('--total_steps', default=100001, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--eval_epochs', default=10, type=int)
    parser.add_argument('--batch_size', '-bs', default=128, type=int)
    # save
    parser.add_argument('--config_dir', type=str, default='config')
    parser.add_argument('--save_tb', default=True, action='store_true')
    parser.add_argument('--save_buffer', '-sb', default=False, action='store_true')
    # seed
    parser.add_argument('--seed_list', '-s', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--seed', default=1, type=int)
    # cuda
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--cuda_id', type=int, default=0)
    ####################################################################
    parser.add_argument('--nstep', default=1, type=int)
    parser.add_argument('--nstep_rew', '-nsr', default=5, type=int)
    parser.add_argument('--num_updates', default=1, type=int)
    parser.add_argument('--extr_q_update_freq', '-euf', default=1, type=int)
    parser.add_argument('--actor_mode', default='sum', type=str)
    parser.add_argument('--buf_dtype', default='nstep', type=str)
    parser.add_argument('--critic_lr', default=5e-4, type=float)
    parser.add_argument('--actor_lr', default=5e-4, type=float)
    parser.add_argument('--alpha_lr', default=5e-4, type=float)
    parser.add_argument('--extr_lr', '-elr', default=5e-4, type=float)
    parser.add_argument('--extr_latent_dim', default=50, type=int)
    parser.add_argument('--targ_extr', '-te', default=False, action='store_true')
    parser.add_argument('--num_q', default=2, type=int)
    parser.add_argument('--omega', default=0.0, type=float)
    parser.add_argument('--rs_discount', '-rdis', default=0.8, type=float)
    parser.add_argument('--enable_env_labels', '-enl', default=False, action='store_true')
    parser.add_argument('--rew_up', '-rup', default=False, action='store_true')
    parser.add_argument('--cpc', '-cpc', default=False, action='store_true')
    parser.add_argument('--cl_mode', '-cl', default='clmse_max', type=str)
    parser.add_argument('--sm_mode', '-sm', default='ar10-kcl256', type=str)
    parser.add_argument('--rew_outdim', '-rdim', default=5, type=int)
    ####################################################################
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args


def set_dcs_multisources(domain_name, task_name, image_size, action_repeat,
                         frame_stack, background_dataset_path, num_sources,
                         difficulty='hard', dynamic=True, distract_mode='val', video_start_idxs=None,
                         background=False, camera=False, color=False, num_videos=1, **kargs):
    assert difficulty in _DIFFICULTY
    assert distract_mode in _DISTRACT_MODE
    if video_start_idxs is None:
        video_start_idxs = [num_videos * i for i in range(num_sources)]

    train_envs = []
    for i in range(num_sources):
        background_kwargs, camera_kwargs, color_kwargs = None, None, None

        if background:
            background_kwargs = suite_utils.get_background_kwargs(
                domain_name, num_videos, dynamic, background_dataset_path, distract_mode)
            background_kwargs['start_idx'] = video_start_idxs[i]
        
        if camera:
            if scale is None:
                scale = suite_utils.DIFFICULTY_SCALE[difficulty]
            camera_kwargs = suite_utils.get_camera_kwargs(domain_name, scale, dynamic)
        
        if color:
            if scale is None:
                scale = suite_utils.DIFFICULTY_SCALE[difficulty]
            color_kwargs = suite_utils.get_color_kwargs(scale, dynamic)

        seed = np.random.randint(1, 100000)
        env = utils.make_env(domain_name, task_name, seed, image_size,
                             action_repeat, frame_stack, background_dataset_path,
                             background_kwargs=background_kwargs,
                             camera_kwargs=camera_kwargs, color_kwargs=color_kwargs)
        env.seed(seed)
        train_envs.append(env)
    return train_envs


def collect_data(train_envs, agent, replay_buffer, total_steps):

    avg_ret = []
    env_id, episode, episode_reward, episode_step, done = 0, 0, 0, 0, False
    num_sources = len(train_envs)
    env = train_envs[env_id]
    epoch_start_times = start_time = time.time()
    o = env.reset()
    s = env.current_state
    # import pdb
    # pdb.set_trace()

    for step in range(total_steps + 1):
        # sample action for data collection
        with utils.eval_mode(agent):
            a = agent.select_action(o, deterministic=True)

        o2, r, done, infos = env.step(a)
        s2 = env.current_state
        episode_reward += r
        episode_step += 1
        d_bool = 0 if episode_step == env._max_episode_steps else float(done)
        replay_buffer.add(o, a, r, o2, d_bool, done, episode, episode_step, env_id, None, infos, s, s2)

        o, s = o2, s2

        if done:
            print("Total T: {} Reward: {:.3f} Episode Num: {} Episode T: {} Env ID: {} Times: {}".format(
                step+1, episode_reward, episode, episode_step, env_id, (time.time() - start_time) / 3600))

            avg_ret.append(episode_reward)
            episode += 1
            env_id = episode % num_sources
            env = train_envs[env_id]
            o, done, episode_reward, episode_step = env.reset(), False, 0, 0
            s = env.current_state

    print("AverageTestEpRet: {} StdTestEpRet: {} TotalEpisode Num: {}".format(
        np.mean(avg_ret), np.std(avg_ret), episode))


def train_task_rel(agent, net_tr, net_tir, replay_buffer, epochs, eval_epochs, batch_size, L, TBL, device):
    def rep(agent, o):
        with utils.eval_mode(agent):
            s = agent.extr(agent._handle_data(o))
            return agent.rl.actor.forward_trunk(s)

    cross_entropy_loss = nn.CrossEntropyLoss()
    tr_optimizer = torch.optim.Adam(net_tr.parameters(), lr=0.001)
    tir_optimizer = torch.optim.Adam(net_tir.parameters(), lr=0.001)
    labels = torch.arange(batch_size).long().to(device)
    label_evals = torch.arange(500).long().to(device)
    TR, TIR, ACCTR, ACCTIR = [], [], [], []
    start_time = time.time()
    for epoch in range(epochs):
        for step in range(625):
            batch = replay_buffer.sample_ablate(batch_size)
            obs, state, envl, idx = batch
            latent_s = rep(agent, obs).detach()
            # task-irrelevance
            out_tir = net_tir(latent_s)
            loss_tir = cross_entropy_loss(out_tir, envl.long().squeeze(-1))
            acc_tir = (out_tir.max(dim=-1)[1] == envl.long().squeeze(-1)).sum(-1).detach() / envl.size(0)
            utils.update_params(tir_optimizer, loss_tir)
            # task-relevance
            out_tr = net_tr(latent_s)
            with torch.no_grad():
                state[state == 0.0] = 0.0001
                s_norm = state / torch.norm(state, dim=-1, p=2, keepdim=True)
            sim = (out_tr / torch.norm(out_tr, dim=-1, p=2, keepdim=True)) @ s_norm.t()
            sim /= 0.1
            pred_prob = torch.softmax(sim, dim=-1)
            target = F.one_hot(labels, sim.size(1)).to(device)
            diff = pred_prob - target.float()
            loss_tr = (sim * diff).sum(-1).mean(-1)
            acc_tr = (pred_prob.max(dim=-1)[1] == labels).sum(-1) / labels.size(0)
            utils.update_params(tr_optimizer, loss_tr)

            record_tr = cross_entropy_loss(sim, labels).detach()
            # print info
            if (step+1) % 125 == 0:
                print("| Epoch {} | Step {} | LossTR {:.3f} | AccTR {:.3f}% | LossTIR {:.3f} | AccTIR {:.3f}% | Times {:.3f}".format(
                    epoch, step+1, record_tr, acc_tr*100, loss_tir, acc_tir*100, (time.time() - start_time) / 3600
                ))
            L.store(LossTR=record_tr, AccTR=acc_tr, LossTIR=loss_tir, AccTIR=acc_tir)

            step = epoch * 625 + step
            TBL.log('train/LossTR', record_tr, step)
            TBL.log('train/AccTR', acc_tr, step)
            TBL.log('train/LossTIR', loss_tir, step)
            TBL.log('train/AccTIR', acc_tir, step)

        # evaluation
        if (epoch+1) % eval_epochs == 0:
            avg_loss_tr, avg_loss_tir = [], []
            avg_acc_tr, avg_acc_tir = [], []
            with torch.no_grad():
                for step in range(40):
                    batch = replay_buffer.sample_val_ablate()
                    o, s, el, idx = batch
                    z = rep(agent, o)
                    # task-irrelevance
                    out_tir = net_tir(z)
                    loss_tir = cross_entropy_loss(out_tir, el.long().squeeze(-1))
                    acc_tir = (out_tir.max(dim=-1)[1] == el.long().squeeze(-1)).sum(-1) / el.size(0)
                    # task-relevance
                    s[s == 0.0] = 0.0001
                    out_tr = net_tr(z)
                    sim = utils.cosine_similarity(out_tr, s)
                    loss_tr = cross_entropy_loss(sim / 0.1, label_evals)
                    acc_tr = (sim.max(dim=-1)[1]==label_evals).sum(-1) / label_evals.size(0)
                    avg_loss_tr.append(loss_tr.data.cpu().numpy())
                    avg_loss_tir.append(loss_tir.data.cpu().numpy())
                    avg_acc_tr.append(acc_tr.cpu().numpy())
                    avg_acc_tir.append(acc_tir.cpu().numpy())

            TBL.log(f'eval/mean_loss_tr', np.mean(avg_loss_tr), epoch)
            TBL.log(f'eval/std_loss_tr', np.std(avg_loss_tr), epoch)
            TBL.log(f'eval/mean_acc_tr', np.mean(avg_acc_tr), epoch)

            TBL.log(f'eval/mean_loss_tir', np.mean(avg_loss_tir), epoch)
            TBL.log(f'eval/std_loss_tir', np.std(avg_loss_tir), epoch)
            TBL.log(f'eval/mean_acc_tir', np.mean(avg_acc_tir), epoch)

            L.log_tabular('Epoch', epoch)
            L.log_tabular('LossTR', average_only=True)
            L.log_tabular('AccTR', average_only=True)
            L.log_tabular('LossTIR', average_only=True)
            L.log_tabular('AccTIR', average_only=True)
            L.log_tabular('TestTR', np.mean(avg_loss_tr))
            L.log_tabular('TestAccTR', np.mean(avg_acc_tr))
            L.log_tabular('TestTIR', np.mean(avg_loss_tir))
            L.log_tabular('TestAccTIR', np.mean(avg_acc_tir))
            L.log_tabular('Time', (time.time() - start_time)/3600)
            L.dump_tabular()

            TR.append(np.mean(avg_loss_tr))
            TIR.append(np.mean(avg_loss_tir))
            ACCTR.append(np.mean(avg_acc_tr))
            ACCTIR.append(np.mean(avg_acc_tir))
    return np.array(TR), np.array(TIR), np.array(ACCTR), np.array(ACCTIR)


def main(args, device, work_dir, config):
    if args.seed == -1: 
        args.__dict__["seed"] = np.random.randint(1, 1000000)
    utils.set_seed_everywhere(args.seed)
    _, domain_name, task_name = args.env.split('.')

    # make directory
    work_dir /= 'dataset'
    data_dir = utils.make_dir(work_dir / f'{args.env}')
    name = 'dmc.cheetah.run-Ba2-b256-cresp-sum-e50-r10-k256-bns5-g80'
    exp_name = Path.cwd() / name / f'{name}_s{args.seed}'
    load_module_dir = exp_name + '/model'
    # Setting Environment
    config['buffer_params']['capacity'] = 200000
    config['setting']['distract_mode'] ='val'
    envs = set_dcs_multisources(domain_name, task_name, args.image_size, args.action_repeat, **config['setting'])

    obs_shape = (9, args.image_size, args.image_size)
    state_shape = envs[0].state_space.shape
    action_shape = envs[0].action_space.shape
    action_limit = envs[0].action_space.high[0]

    replay_buffer = ReplayBuffer(obs_shape=obs_shape,
                                 state_shape=state_shape,
                                 action_shape=action_shape,
                                 buffer_dir=data_dir,
                                 batch_size=args.batch_size,
                                 device=device,
                                 **config['buffer_params'])

    config.update(dict(obs_shape=obs_shape, batch_size=256, device=device))
    config['agent_base_params']['num_sources'] = args.num_sources
    config['agent_base_params']['action_repeat'] = args.action_repeat
    config['algo_params'].update(dict(action_shape=action_shape,
                                      action_limit=action_limit,
                                      device=device))

    agent = init_agent(args.agent, config)
    start_time = time.time()
    step = 500000 // args.action_repeat
    print(f'Load Dir = {load_module_dir}, Step = {step}')
    # agent.load(load_module_dir, step)
    agent.extr.load_state_dict(
        torch.load('%s/extr_%s.pt' % (load_module_dir, step), map_location=torch.device('cpu'))
    )
    agent.rl.actor.load_state_dict(
        torch.load('%s/actor_%s.pt' % (load_module_dir, step), map_location=torch.device('cpu'))
    )

    if args.save_buffer:
        collect_data(envs, agent,replay_buffer, args.total_steps)
    else:
        logger_kwargs = setup_logger_kwargs('cr-cresp', args.seed, work_dir)
        L = EpochLogger(**logger_kwargs)
        L.save_config(config)

        exp_dir = work_dir / 'cr-cresp' / f'cr-cresp_s{args.seed}'
        TBL = Logger(exp_dir, action_repeat=args.action_repeat, use_tb=args.save_tb)

        replay_buffer.load(data_dir)
        print("The capacity of the replay buffer now is {}".format(replay_buffer.size))
        assert replay_buffer.size == replay_buffer.idx
        replay_buffer.shrank(20000)

        net_tr = utils.mlp(50, 256, state_shape[0], 2).to(device)
        net_tir = utils.mlp(50, 256, 20, 2).to(device)
        print('NET_TR', net_tr)
        print('NET_TIR', net_tir)

        TR, TIR, ACCTR, ACCTIR = train_task_rel(
            agent, net_tr, net_tir, replay_buffer, args.epochs,
            args.eval_epochs, args.batch_size, L, TBL, device
        )
        np.save(str(exp_dir / 'tr'), TR)
        np.save(str(exp_dir / 'tir'), TIR)
        np.save(str(exp_dir / 'acc_tr'), ACCTR)
        np.save(str(exp_dir / 'acc_tir'), ACCTIR)
        print("| Low TestTR {} | Low TestTIR {} |".format(TR.min(), TIR.min()))

    for env in envs:
        env.close()


if __name__ == '__main__':

    args = parse_args()
    cuda_id = f"cuda:{args.cuda_id}"
    device = torch.device(cuda_id if args.cuda else "cpu")
    if args.save_buffer:
        work_dir = Path.cwd()
        config = utils.read_config(args, work_dir / args.config_dir)
        main(args, device, work_dir, config)
    else:
        for i in args.seed_list:
            work_dir = Path.cwd()
            config = utils.read_config(args, work_dir / args.config_dir)
            torch.multiprocessing.set_start_method('spawn', force=True)
            args.seed, config['setting']['seed'] = i, i
            main(args, device, work_dir, config)
