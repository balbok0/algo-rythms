import numpy as np

import os
import shutil
import torch

from datetime import datetime, tzinfo
from pathlib import Path


class Logger():
    def __init__(self):
        self.time_stamp = str(datetime.now())
        self.time_stamp = self.time_stamp.replace(':', '')
        self.iter = 0
        self.log_folder = Path('log') / self.time_stamp

        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)

    def save_end(self, model, kwargs):
        torch.save(model.state_dict(), self.log_folder / 'model.pth')
        for k, v in kwargs.items():
            np.save(self.log_folder / k, v)

    def save_gan_iter(self, model_generator, model_adversary, **kwargs):
        self.iter += 1
        torch.save(
            model_generator.state_dict(),
            self.log_folder / 'iter_{}_model_generator.pth'.format(self.iter)
        )
        torch.save(
            model_adversary.state_dict(),
            self.log_folder / 'iter_{}_model_adversary.pth'.format(self.iter)
        )
        for k, v in kwargs.items():
            np.save(self.log_folder / 'iter_{}_{}'.format(self.iter, k), v)

    def save_gan_end(self, model_generator, model_adversary, **kwargs):
        torch.save(
            model_generator.state_dict(),
            self.log_folder / 'final_model_generator.pth'
        )
        torch.save(
            model_adversary.state_dict(),
            self.log_folder / 'final_model_adversary.pth'
        )
        for k, v in kwargs.items():
            np.save(self.log_folder / 'final_{}'.format(k), v)

    def clean(self):
        shutil.rmtree(self.log_folder)
