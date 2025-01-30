'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from absl import app, flags
from models import *
from utils import get_moments_dict, save_layer_histogram_plots, load_config, maybe_make_folders_for_model
from dataloaders import get_dataloaders

flags.DEFINE_string('config', 'config/config.yaml', 'Path to config.yaml file.')
flags.DEFINE_integer('job_idx', None, 'Index of the hyperparameter combination to use.')
FLAGS = flags.FLAGS

def main(_):
    CFG_PATH, JOB_IDX = FLAGS.config, FLAGS.job_idx
    cfg, _ = load_config(CFG_PATH, JOB_IDX)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(torch.cuda.is_available())
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA current device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(device)



    """
    from forked repo
    chaging to cleaner config structure
    """
    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()
    # net = RegNetX_200MF()
    # net = SimpleDLA()


    if cfg.model == "resnet18":
        net = ResNet18()
    elif cfg.model == "vgg19":
        net = VGG('VGG19')
    elif cfg.model == "mobilenet":
        net = MobileNet()
    elif cfg.model == "lenet":
        net = GoogLeNet()
    elif cfg.model == "simplelda":
        net = SimpleDLA()
    elif cfg.model == "efficientnet":
        net = EfficientNetB0()
    elif cfg.model == "senet18":
        net = SENet18()


    trainloader, testloader = get_dataloaders(batch_size=cfg.batch_size)

    net = net.to(device)
    maybe_make_folders_for_model(net, cfg.plot_dir)
    # add model name in join 
    plot_path = os.path.join(cfg.plot_dir, net.__class__.__name__) 



    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), 
        eps=cfg.eps, weight_decay=cfg.weight_decay, fused=cfg.fused_optim) 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    # Training
    def train(epoch):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(f"Epoch: {epoch}, Loss: {train_loss / len(trainloader) }")

    """
    for epoch in range(cfg.epochs):
        train(epoch)

        if epoch % cfg.log_every == 0:
            moments_dict = get_moments_dict(net, optimizer)

            for layer_name in moments_dict.keys():
                if layer_name.startswith('module.'):
                    layer_name = layer_name[len('module.'):]
                    print(layer_name)
                os.makedirs(os.path.join(plot_path, layer_name), exist_ok=True)
                save_path = os.path.join(plot_path, layer_name)
                save_layer_histogram_plots(epoch, moments_dict, layer_name, savepath = save_path)
        

        scheduler.step()
    """
    for epoch in range(cfg.epochs):
        train(epoch)

        if epoch % cfg.log_every == 0:
            moments_dict = get_moments_dict(net, optimizer)

            for layer_name in moments_dict.keys():
                if layer_name.startswith('module.'):
                    layer_name = layer_name[len('module.'):]
                    print(layer_name)
                os.makedirs(os.path.join(plot_path, layer_name), exist_ok=True)
                save_path = os.path.join(plot_path, layer_name)
                save_layer_histogram_plots(epoch, moments_dict, layer_name, savepath=save_path)

        scheduler.step()
"""
    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc
        """



if __name__ == '__main__':
    app.run(main)