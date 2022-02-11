import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


class BReplayBuffer(Dataset):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, buffer_dir, batch_size, nstep,
                 nstep_rew, gamma, lambd, device, save_buffer, image_size=84, **kwargs):
        self.capacity = capacity
        self.buffer_dir = buffer_dir
        self.batch_size = batch_size
        self.nstep = nstep
        self.nstep_rew = nstep_rew
        self.gamma = gamma
        self.lambd = lambd
        self.device = device
        self.save_buffer = save_buffer
        self.image_size = image_size
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.logp_a = np.empty((capacity, 1), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.env_labels = np.empty((capacity, 1), dtype=np.float32)
        self.obs_number = np.empty((capacity, 1), dtype=np.float32)

        # save the start point and end point of an episode
        self.ep_start_idxs = np.zeros((capacity,), dtype=np.int)
        self.ep_end_idxs = np.zeros((capacity,), dtype=np.int)
        self.ep_end_idxs_rew = np.zeros((capacity,), dtype=np.int)
        # save episode num which can be used to update the agent
        self.ep_num_list = []

        self.idx = 0
        self.ep_num = 0
        self.anchor_idx = 0
        self.last_save = 0
        self.full = False
        self.valid_start_ep_num = 0
        self.infos = {}

    @property
    def size(self):
        return self.capacity if self.full else self.idx

    def minus(self, y, x):
        return y - x if y >= x else y - x + self.capacity

    def add_obs(self, obs, ep_num):
        np.copyto(self.obses[self.idx : self.idx + 2], np.stack((obs[:3], obs[3:6]), axis=0))
        self.idx = (self.idx + 2) % self.capacity
        self.ep_start_idxs[self.ep_num] = self.idx

    def add(self, obs, action, reward, next_obs, done, d, ep_num, ep_len, env_label=0, logp_a=None, infos=None):
        if self.full and self.anchor_idx == self.idx:
            if self.ep_end_idxs[self.valid_start_ep_num] == self.ep_start_idxs[self.valid_start_ep_num]:
                del self.ep_num_list[0]
                self.valid_start_ep_num = self.ep_num_list[0]
                self.anchor_idx = self.minus(self.ep_start_idxs[self.valid_start_ep_num], 2)
            else:
                self.anchor_idx = (self.anchor_idx + 1) % self.capacity
                self.ep_start_idxs[self.valid_start_ep_num] = (self.anchor_idx + 2) % self.capacity

        np.copyto(self.obses[self.idx], obs[-3:])
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.not_dones[self.idx], not done)
        np.copyto(self.env_labels[self.idx], env_label)
        np.copyto(self.obs_number[self.idx], self.idx)
        if logp_a is not None:
            np.copyto(self.logp_a[self.idx], logp_a)

        if d:
            self.idx = (self.idx + 1) % self.capacity
            np.copyto(self.obses[self.idx], next_obs[-3:])
            # nstep = self.nstep - 1
            nstep = self.nstep - 1 if self.nstep > 1 else 1 # prepare for predictor
            nstep_rew = self.nstep_rew - 1 if self.nstep_rew > 1 else 1
            self.ep_end_idxs[self.ep_num] = self.minus(self.idx, nstep)
            self.ep_end_idxs_rew[self.ep_num] = self.minus(self.idx, nstep_rew)
            self.ep_num_list.append(self.ep_num)
            if self.save_buffer:
                self.save(self.buffer_dir, self.idx+1, self.ep_num, ep_len)
            self.ep_num += 1

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def get_rew(self, obs_idxs, idxs, agent=None):
        gamma = 1.0
        if self.nstep == 1:
            actions2 = torch.as_tensor(self.actions[(idxs + 1) % self.capacity], device=self.device)
            rewards2 = torch.as_tensor(self.rewards[(idxs + 1) % self.capacity], device=self.device)
            return torch.as_tensor(self.rewards[idxs], device=self.device), dict(gamma=gamma, nstep=self.nstep, act2=actions2, rew2=rewards2)
        # nstep
        rewards = np.zeros((idxs.shape[0], 1))
        for i in range(self.nstep):
            step_rewards = self.rewards[(idxs + i) % self.capacity]
            rewards += gamma * step_rewards
            gamma *= self.gamma
        return torch.as_tensor(rewards, device=self.device), dict(gamma=gamma, nstep=self.nstep)

    def sample_image_fs3_idxs(self, ep_idxs, start, end):
        end[start >= end] += self.capacity
        idxs = np.random.randint(start, end)
        obs_idxs = idxs.repeat(3).reshape(-1, 3)
        obs_idxs[:, 0] -= 2
        obs_idxs[:, 1] -= 1
        obs_idxs = obs_idxs.reshape(-1)
        obs_idxs[obs_idxs < 0] += self.capacity
        return idxs % self.capacity, obs_idxs % self.capacity

    def sample(self, batch_size=None, agent=None):
        if batch_size is None:
            batch_size = self.batch_size

        ep_idxs = np.random.choice(np.array(self.ep_num_list), batch_size)
        ep_start_idxs = self.ep_start_idxs[ep_idxs]
        ep_end_idxs = self.ep_end_idxs[ep_idxs]
        idxs, obs_idxs = self.sample_image_fs3_idxs(ep_idxs, ep_start_idxs, ep_end_idxs)

        obses = self.obses[obs_idxs].reshape(-1, 9, *self.obses.shape[-2:])
        next_obses = self.obses[(obs_idxs + self.nstep) % self.capacity].reshape(-1, 9, *self.obses.shape[-2:])

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards, infos = self.get_rew(obs_idxs, idxs, agent)
        next_obses = torch.as_tensor(next_obses, device=self.device).float()
        not_dones = torch.as_tensor(self.not_dones[(idxs + self.nstep - 1) % self.capacity], device=self.device).float()
        env_labels = torch.as_tensor(self.env_labels[idxs], device=self.device)
        obs_num = torch.as_tensor(self.obs_number[idxs], device=self.device)

        transition = dict(obs=obses, obs2=next_obses, act=actions, rew=rewards.float(),
                          not_done=not_dones, env_labels=env_labels, obs_num=obs_num, infos=infos)
        return transition

    def sample_obs(self, batch_size=None, enable_labels=False):
        if batch_size is None:
            batch_size = self.batch_size
        
        ep_idxs = np.random.choice(np.array(self.ep_num_list), batch_size)
        ep_start_idxs = self.ep_start_idxs[ep_idxs]
        ep_end_idxs = self.ep_end_idxs[ep_idxs]
        idxs, obs_idxs = self.sample_image_fs3_idxs(ep_start_idxs, ep_end_idxs)

        obses = self.obses[obs_idxs].reshape(-1, 9, *self.obses.shape[-2:])
        obses = torch.as_tensor(obses, device=self.device).float()
        if enable_labels:
            env_labels = torch.as_tensor(self.env_labels[idxs], device=self.device)
            return obses, env_labels
        return obses

    def sample_traj(self, batch_size=None, agent=None):
        if batch_size is None:
            batch_size = self.batch_size
        ep_idxs = np.random.choice(np.array(self.ep_num_list), batch_size)
        ep_start_idxs = self.ep_start_idxs[ep_idxs]
        ep_end_idxs = self.ep_end_idxs_rew[ep_idxs]
        idxs, obs_idxs = self.sample_image_fs3_idxs(ep_idxs, ep_start_idxs, ep_end_idxs)

        traj_o, traj_a, traj_r, traj_loga = [], [], [], []
        for i in range(self.nstep_rew):
            obs = self.obses[(obs_idxs + i) % self.capacity].reshape(-1, 9, *self.obses.shape[-2:])
            traj_o.append(torch.as_tensor(obs, device=self.device).float())
            act = self.actions[(idxs + i) % self.capacity]
            traj_a.append(torch.as_tensor(act, device=self.device))
            step_rewards = self.rewards[(idxs + i) % self.capacity]
            traj_r.append(torch.as_tensor(step_rewards, device=self.device))
            logp_a = self.logp_a[(idxs + i) % self.capacity]
            traj_loga.append(torch.as_tensor(logp_a, device=self.device))

        traj_o = torch.stack(traj_o, dim=0) # (nstep, batch_size, 9, img_size, img_size)
        traj_a = torch.stack(traj_a, dim=0) # (nstep, batch_size, act_dim)
        traj_loga = torch.stack(traj_loga, dim=0).squeeze(-1) # (nstep, batch_size)
        traj_r = torch.stack(traj_r, dim=0).squeeze(-1) # (nstep, batch_size)
        o_end = self.obses[(obs_idxs + self.nstep_rew) % self.capacity].reshape(-1, 9, *self.obses.shape[-2:])
        o_end = torch.as_tensor(o_end, device=self.device).float()
        envl = torch.as_tensor(self.env_labels[idxs], device=self.device)
        obs_num = torch.as_tensor(self.obs_number[idxs], device=self.device)
        traj = dict(traj_o=traj_o, traj_a=traj_a, traj_log_pa=traj_loga, traj_r=traj_r, obs_end=o_end, env_labels=envl, obs_num=obs_num)
        return traj

    def sample_abst_traj(self, batch_size=None, agent=None):
        if batch_size is None:
            batch_size = self.batch_size
        ep_idxs = np.random.choice(np.array(self.ep_num_list), batch_size)
        ep_start_idxs = self.ep_start_idxs[ep_idxs]
        ep_end_idxs = self.ep_end_idxs_rew[ep_idxs]
        idxs, obs_idxs = self.sample_image_fs3_idxs(ep_idxs, ep_start_idxs, ep_end_idxs)

        traj_a, traj_r, traj_loga = [], [], []
        obs = self.obses[obs_idxs].reshape(-1, 9, *self.obses.shape[-2:])
        obs = torch.as_tensor(obs, device=self.device).float()
        for i in range(self.nstep_rew):
            act = self.actions[(idxs + i) % self.capacity]
            traj_a.append(torch.as_tensor(act, device=self.device))
            step_rewards = self.rewards[(idxs + i) % self.capacity]
            traj_r.append(torch.as_tensor(step_rewards, device=self.device))
            logp_a = self.logp_a[(idxs + i) % self.capacity]
            traj_loga.append(torch.as_tensor(logp_a, device=self.device))

        traj_a = torch.stack(traj_a, dim=0) # (nstep, batch_size, act_dim)
        traj_loga = torch.stack(traj_loga, dim=0).squeeze(-1) # (nstep, batch_size)
        traj_r = torch.stack(traj_r, dim=0).squeeze(-1) # (nstep, batch_size)
        o_end = self.obses[(obs_idxs + self.nstep_rew) % self.capacity].reshape(-1, 9, *self.obses.shape[-2:])
        o_end = torch.as_tensor(o_end, device=self.device).float()
        envl = torch.as_tensor(self.env_labels[idxs], device=self.device)
        obs_num = torch.as_tensor(self.obs_number[idxs], device=self.device)
        traj = dict(traj_o=obs, traj_a=traj_a, traj_log_pa=traj_loga, traj_r=traj_r, obs_end=o_end, env_labels=envl, obs_num=obs_num)
        return traj

    def save(self, save_dir, idx, ep_num, ep_len):
        if idx == self.last_save:
            return
        _idx = idx + self.capacity if idx < self.last_save else idx
        path = save_dir / '%d_%d_from%d_to%d.pt' % (self.ep_num, ep_len, self.last_save, _idx)
        payload = [
            self.obses[self.last_save:idx] if _idx == idx else np.concatenate((self.obses[self.last_save:], self.obses[:idx%self.capacity]), axis=0),
            self.actions[self.last_save:idx] if _idx == idx else np.concatenate((self.actions[self.last_save:], self.actions[:idx%self.capacity]), axis=0),
            self.rewards[self.last_save:idx] if _idx == idx else np.concatenate((self.rewards[self.last_save:], self.rewards[:idx%self.capacity]), axis=0),
            self.logp_a[self.last_save:idx] if _idx == idx else np.concatenate((self.logp_a[self.last_save:], self.logp_a[:idx%self.capacity]), axis=0),
            self.not_dones[self.last_save:idx] if _idx == idx else np.concatenate((self.not_dones[self.last_save:], self.not_dones[:idx%self.capacity]), axis=0),
            self.env_labels[self.last_save:idx] if _idx == idx else np.concatenate((self.env_labels[self.last_save:], self.env_labels[:idx%self.capacity]), axis=0),
        ]
        self.last_save = idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            ep_num, ep_len, start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = save_dir / chunk
            payload = torch.load(path)
            end = (end - 1 + self.capacity) % self.capacity + 1
            if end < start:
                self.obses[start:], self.obses[:end] = payload[0][:self.capacity-start], payload[0][self.capacity-start:]
                self.actions[start:], self.actions[:end] = payload[1][:self.capacity-start], payload[1][self.capacity-start:]
                self.rewards[start:], self.rewards[:end] = payload[2][:self.capacity-start], payload[2][self.capacity-start:]
                self.logp_a[start:], self.logp_a[:end] = payload[3][:self.capacity-start], payload[3][self.capacity-start:]
                self.not_dones[start:], self.not_dones[:end] = payload[4][:self.capacity-start], payload[4][self.capacity-start:]
                self.env_labels[start:], self.env_labels[:end] = payload[5][:self.capacity-start], payload[5][self.capacity-start:]
            else:
                self.obses[start:end] = payload[0]
                self.actions[start:end] = payload[1]
                self.rewards[start:end] = payload[2]
                self.logp_a[start:end] = payload[3]
                self.not_dones[start:end] = payload[4]
                self.env_labels[start:end] = payload[5]
            self.ep_num_list.append(ep_num)
            self.ep_start_idxs[ep_num] = (start + 2) % self.capacity
            self.ep_end_idxs[ep_num] = self.minus(end, self.nstep)
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(0, self.size, size=1)
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        logp_a = self.logp_a[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]
        env_labels = self.env_labels[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done, env_labels

    def __len__(self):
        return self.capacity

