from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import json
import os
import shutil
import torch
import torchvision
import numpy as np
from termcolor import colored

FORMAT_CONFIG = {
    'rl': {
        'train': [
            ('episode', 'E', 'int'), ('step', 'S', 'int'),
            ('duration', 'D', 'time'), ('episode_reward', 'R', 'float'),
            ('actor_loss', 'A_LOSS', 'float'), ('critic_loss', 'CR_LOSS', 'float')
        ],
        'eval': [('step', 'S', 'int'), ('episode_reward', 'ER', 'float')]
    }
}


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, file_name, formating):
        self._file_name = file_name
        if os.path.exists(file_name):
            os.remove(file_name)
        self._formating = formating
        self._meters = defaultdict(AverageMeter)

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith('train'):
                key = key[len('train') + 1:]
            else:
                key = key[len('eval') + 1:]
            key = key.replace('/', '_')
            data[key] = meter.value()
        return data

    def _dump_to_file(self, data):
        with open(self._file_name, 'a') as f:
            f.write(json.dumps(data) + '\n')

    def _format(self, key, value, ty):
        template = '%s: '
        if ty == 'int':
            template += '%d'
        elif ty == 'float':
            template += '%.04f'
        elif ty == 'time':
            template += '%.01f s'
        else:
            raise 'invalid format type: %s' % ty
        return template % (key, value)

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, 'yellow' if prefix == 'train' else 'green')
        pieces = ['{:5}'.format(prefix)]
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print('| %s' % (' | '.join(pieces)))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data['step'] = step
        self._dump_to_file(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir, use_tb=True, action_repeat=2, config='rl'):
        self._log_dir = log_dir
        self.action_repeat = action_repeat
        if use_tb:
            tb_dir = os.path.join(log_dir, 'tb')
            if os.path.exists(tb_dir):
                shutil.rmtree(tb_dir)
            self._sw = SummaryWriter(tb_dir)
        else:
            self._sw = None
        self._train_mg = MetersGroup(
            os.path.join(log_dir, 'train.log'),
            formating=FORMAT_CONFIG[config]['train']
        )
        self._eval_mg = MetersGroup(
            os.path.join(log_dir, 'eval.log'),
            formating=FORMAT_CONFIG[config]['eval']
        )

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def _try_sw_log_image(self, key, image, step):
        if self._sw is not None:
            assert image.dim() == 3, print(image.dim())
            grid = torchvision.utils.make_grid(image.unsqueeze(1))
            self._sw.add_image(key, grid, step)

    def _try_sw_log_video(self, key, frames, step):
        if self._sw is not None:
            frames = torch.from_numpy(np.array(frames))
            frames = frames.unsqueeze(0)
            self._sw.add_video(key, frames, step, fps=30)

    def _try_sw_log_histogram(self, key, histogram, step):
        if self._sw is not None:
            self._sw.add_histogram(key, histogram, step)

    def save_log(self, info_dict, info_name, step, action_repeat=None):
        for key in info_dict.keys():
            if not isinstance(info_dict[key], float):
                info_dict[key] = info_dict[key].mean().item()
            self.log(f'{info_name}/{key}', info_dict[key], step, action_repeat)

    def save_log_param(self, info_dict, info_name, step, action_repeat=None):
        for key, values in info_dict.items():
            if not isinstance(info_dict[key], float):
                assert values.ndim < 2, print(info_name, key, values.ndim)
            self.log_histogram(f'{info_name}/{key}', values, step, action_repeat)

    def save_log_imgs(self, info_dict, info_name, step, action_repeat=None):
        for key, values in info_dict.items():
            if not isinstance(info_dict[key], float):
                if values.ndim == 4:
                    values = values.mean(0)
                assert values.ndim == 3, print(info_name, key, values.ndim)
            self.log_image(f'{info_name}/{key}', values, step, action_repeat)

    def log(self, key, value, step, action_repeat=None, n=1):
        # assert 'train' in key or 'eval' in key
        if action_repeat is None:
            action_repeat = self.action_repeat
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value / n, (step + 1) * action_repeat)
        mg = self._train_mg if key.startswith('train') else self._eval_mg
        mg.log(key, value, n)

    def log_param(self, key, param, step, action_repeat=None):
        self.log_histogram(key + '_w', param.weight.data, step, action_repeat)
        if hasattr(param.weight, 'grad') and param.weight.grad is not None:
            self.log_histogram(key + '_w_g', param.weight.grad.data, step, action_repeat)
        if hasattr(param, 'bias') and param.bias is not None:
            self.log_histogram(key + '_b', param.bias.data, step, action_repeat)
            if hasattr(param.bias, 'grad') and param.bias.grad is not None:
                self.log_histogram(key + '_b_g', param.bias.grad.data, step, action_repeat)

    def log_image(self, key, image, step, action_repeat=None):
        # assert 'train' in key or 'eval' in key
        if action_repeat is None:
            action_repeat = self.action_repeat
        self._try_sw_log_image(key, image, (step + 1) * action_repeat)

    def log_images(self, key, image, step, action_repeat=None):
        if action_repeat is None:
            action_repeat = self.action_repeat
        if self._sw is not None:
            if image.dim() == 3 or image.ndim() == 3:
                image = image.view(-1, 3, *image.size()[1:])
            if (image > 1.0).any():
                image /= 255.0
            if (image < 0.0).any():
                image += 0.5
            self._sw.add_images(key, image, (step + 1) * action_repeat)

    def log_video(self, key, frames, step):
        assert 'train' in key or 'eval' in key
        self._try_sw_log_video(key, frames, step)

    def log_histogram(self, key, histogram, step, action_repeat=None):
        if action_repeat is None:
            action_repeat = self.action_repeat
        step = (step + 1) * action_repeat
        assert 'train' in key or 'eval' in key
        self._try_sw_log_histogram(key, histogram, step)

    def dump(self, step):
        self._train_mg.dump(step, 'train')
        self._eval_mg.dump(step, 'eval')
