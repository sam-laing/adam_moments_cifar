'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
import torch

import matplotlib.pyplot as plt
import wandb
from collections import namedtuple
from itertools import product
import yaml

def load_config(path, job_idx=None):
  """
  Parse a yaml file and return the correspondent config as a namedtuple.
  If the config files has multiple entries, returns the one corresponding to job_idx.
  """
  
  with open(path, 'r') as file:
    config_dict = yaml.safe_load(file)
  Config = namedtuple('Config', config_dict.keys())

  if job_idx is None:
    cfg = config_dict
    sweep_size = 1

  else:
    keys = list(config_dict.keys())
    values = [val if isinstance(val, list) else [val] for val in config_dict.values()]
    combinations = list(product(*values))

    sweep_size = len(combinations)
    if job_idx >= sweep_size:
      raise ValueError("job_idx exceeds the total number of hyperparam combinations.")

    combination = combinations[job_idx]
    cfg = {keys[i]: combination[i] for i in range(len(keys))}
  
  return Config(**cfg), sweep_size


def init_wandb(cfg):
    """Initalizes a wandb run"""
    #os.environ["WANDB_API_KEY"] = cfg.wandb_api_key
    os.environ["WANDB__SERVICE_WAIT"] = "600"
    os.environ["WANDB_SILENT"] = "true"
    wandb_run_name = f"{cfg.optim}, lr={cfg.lr}, eps={cfg.eps}, wd={cfg.weight_decay}, b1={cfg.beta1}, b2={cfg.beta2}"
    wandb.init(
        project=cfg.wandb_project, 
        name=wandb_run_name, 
        dir=cfg.wandb_dir,
        config=cfg._asdict()
    )


def get_moments_dict(model, optimizer) -> dict:
    """  
    Returns a dictionary of the first and second moments of the optimizer's 
    moving averages for each layer
    """
    param_to_name = {id(param): name for name, param in model.named_parameters()}

    # Dictionary to store exp_avg and exp_avg_sq
    moments_dict = {}

    # Iterate through the parameter groups in the optimizer
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is None:
                continue
            state = optimizer.state[p]
            if 'exp_avg' in state and 'exp_avg_sq' in state:
                layer_name = param_to_name.get(id(p), "Unknown Layer")
                moments_dict[layer_name] = {
                    'exp_avg': state['exp_avg'].cpu().numpy(),
                    'exp_avg_sq': state['exp_avg_sq'].cpu().numpy()
                }

    return moments_dict

def save_layer_histogram_plots(epoch, moments_dict, layer_name, savepath = "/fast/slaing/exp/cifar_experiments/plots" ):
    """
    Given the dictionary of moments for each layer, plot a histogram of the 
    exp_avg and exp_avg_sq for the specified layer and save plots in the correct folder
    """
    # Get the exp_avg and exp_avg_sq for the specified layer
    layer_moments = moments_dict.get(layer_name, None)
    if layer_moments is None:
        print(f"Layer {layer_name} not found in moments_dict")
        return

    # Get the exp_avg and exp_avg_sq
    exp_avg = layer_moments['exp_avg']
    exp_avg_sq = layer_moments['exp_avg_sq']

    # plot the exp_avg as a histogram
    plt.hist(exp_avg.flatten(), bins=50, alpha=0.95, edgecolor='black', color='#1f77b4')
    plt.title(f"epoch {epoch} exp_avg")
    # save the plot
    plt.savefig(f"exp_avg_{layer_name}.png")

    plt.hist(exp_avg_sq.flatten(), bins=50, alpha=0.95, edgecolor='black', color='#1f77b4')
    plt.title(f"epoch {epoch} exp_avg_sq")
    plt.savefig(f"exp_avg_sq_{layer_name}.png")


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


