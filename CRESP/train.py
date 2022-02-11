import numpy as np
import torch
import time
from collections import deque

import common.utils as utils
# from common import logger


def evaluate(test_env, agent, L, TBL, video, num_eval_episodes, start_time, action_repeat, step):

    def eval(num, env, agent, L, TBL, video, num_episodes, step, start_time, action_repeat):
        all_ep_rewards = []
        all_ep_length = []
        # loop num_episodes
        for episode in range(num_episodes):
            obs = env.reset()
            video.init(enabled=(episode == 0))
            done, episode_reward, episode_length = False, 0, 0
            # evaluate once
            while not done:
                with utils.eval_mode(agent):
                    action = agent.select_action(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                video.record(env)
                episode_reward += reward
                episode_length += 1

            video.save(f'{step * action_repeat}-Env{num}.mp4')
            test_info = {f"TestEpRet{num}": episode_reward, f"TestEpLen{num}": episode_length}
            L.store(**test_info)
            TBL.log('eval/episode_reward', episode_reward, step)
            # record the score
            all_ep_rewards.append(episode_reward)
            all_ep_length.append(episode_length)

        mean, std, best = np.mean(all_ep_rewards), np.std(all_ep_rewards), np.max(all_ep_rewards)
        TBL.log(f'eval/mean_episode_reward{num}', mean, step)
        TBL.log(f'eval/std_episode_reward{num}', std, step)
        TBL.log(f'eval/best_episode_reward{num}', best, step)

    if isinstance(test_env, list):
        # test_env is a list
        for num, t_env in enumerate(test_env):
            eval(num, t_env, agent, L, TBL, video, num_eval_episodes,
                 step, start_time, action_repeat)
    else:
        # test_env is an environment
        eval(0, test_env, agent, L, TBL, video, num_eval_episodes,
             step, start_time, action_repeat)


def train_agent(train_envs, test_env, agent, replay_buffer, L, TBL, video, model_dir,
                total_steps, init_steps, eval_freq, action_repeat, num_updates,
                num_eval_episodes, test, save_model, device, **kwargs):

    env_id, episode, episode_reward, episode_step, best_return, done, logp_a = 0, 0, 0, 0, 0, False, None
    num_sources = len(train_envs)
    env = train_envs[env_id]
    epoch_start_times = start_time = time.time()
    o = env.reset()
    replay_buffer.add_obs(o, episode)

    for step in range(total_steps + 1):

        # sample action for data collection
        if step < init_steps:
            a = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                a, logp_a = agent.select_action(o), None
                if isinstance(a, tuple):
                    a, logp_a = a

        # run training update
        if step >= init_steps and step % num_updates == 0:
            for _ in range(num_updates):
                agent.update(replay_buffer, L, TBL, step+1, step % 100 == 0)

        o2, r, done, infos = env.step(a)
        agent.total_time_steps += 1
        episode_reward += r
        episode_step += 1
        d_bool = 0 if episode_step == env._max_episode_steps else float(done)
        replay_buffer.add(o, a, r, o2, d_bool, done, episode, episode_step, env_id, logp_a, infos)

        # evaluate agent periodically
        if step % eval_freq == 0 and step > init_steps and test:
            TBL.log('eval/episode', episode, step)
            evaluate(test_env, agent, L, TBL, video, num_eval_episodes,
                     start_time, action_repeat, step)
            epoch_fps = eval_freq * action_repeat / (time.time() - epoch_start_times)
            TBL.log('eval/time', (time.time() - start_time) / 3600, step)
            TBL.log('eval/epoch_fps', epoch_fps, step)

            print(f'Update Extr Times: {agent.update_extr_steps} Update Critic Times: {agent.update_critic_steps} Update Actor Times: {agent.update_actor_steps}')
            agent.print_log(L,
                            test_env,
                            (step + 1) // eval_freq,
                            step,
                            action_repeat,
                            test,
                            start_time,
                            epoch_fps)
            epoch_start_times = time.time()

            if save_model:
                agent.save(model_dir, step)

        o = o2

        if done:
            L.store(EpRet=episode_reward, EpLen=episode_step, EpNum=episode)

            TBL.log('train/episode_reward', episode_reward, step)
            TBL.log('train/episode_length', episode_step, step)
            TBL.log('train/episode', episode, step)
            # TBL.dump(step)
            print("Total T: {} Reward: {:.3f} Episode Num: {} Episode T: {}".format(
                agent.total_time_steps, episode_reward, episode, episode_step))
            if best_return < episode_reward:
                best_return = episode_reward
                agent.save(model_dir, f'best_in_env_{env_id}')

            episode += 1
            env_id = episode % num_sources
            env = train_envs[env_id]
            o, done, episode_reward, episode_step = env.reset(), False, 0, 0
            replay_buffer.add_obs(o, episode)
