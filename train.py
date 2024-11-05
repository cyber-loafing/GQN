
import random, io

import numpy as np
import networkx as nx

import torch

from torch.utils.tensorboard import SummaryWriter

from rl_alg.replay import Buffer, PriortizedReplay
from rl_alg.utils import OrnsteinUhlenbeckActionNoise

import pickle, os
import gc
import logging, argparse
import tracemalloc

g_paths = [
    'train_data/copen.pkl',
    'train_data/occupy.pkl',
    # 'train_data/assad.pkl',
    # 'train_data/obama.pkl',
]

syn = False
ratio = 5
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)


random.seed(10)
args = arg_parse()