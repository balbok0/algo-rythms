import numpy as np

import os
import shutil

from datetime import datetime, tzinfo
from pathlib import Path


class Logger():
    def __init__(self):
        self.time_stamp = str(datetime.now())
        self.log_folder = Path('log') / self.time_stamp

        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)

    def save_end(self, **kwargs):
        for k, v in kwargs.items():
            np.save(self.log_folder / k, v)

    def clean(self):
        shutil.rmtree(self.log_folder)
