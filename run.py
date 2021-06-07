import os
import yaml
import torch
import argparse
import logging
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils.logger import ColoredLogger
from datasets.celeba import CelebADataset
from datasets.cub200 import Cub200Dataset
from torch.utils.data import DataLoader
import torchvision.utils as tuitls
from models.DCGAN import DCGAN_Generator, DCGAN_Discriminator


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)


# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--mode', '-m', default = 'train', help = 'the running mode, "train" or "inference"', type = str)
parser.add_argument('--cfg', '-c', default = os.path.join('configs', 'VAE.yaml'), help = 'Config File', type = str)
parser.add_argument('--clean_cache', '-cc', action = 'store_true', help = 'whether to clean the cache of GPU while training, evaluation and testing')
FLAGS = parser.parse_args()
CFG_FILE = FLAGS.cfg
CLEAN_CACHE = FLAGS.clean_cache
MODE = FLAGS.mode

if MODE not in ['train', 'inference']:
    raise AttributeError('mode should be either "train" or "inference".')

with open(CFG_FILE, 'r') as cfg_file:
    cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)
    
model_params = cfg_dict.get('model', {})
g_model_params = model_params.get('generator', {})
d_model_params = model_params.get('discriminator', {})
dataset_params = cfg_dict.get('dataset', {})
optimizer_params = cfg_dict.get('optimizer', {})
g_optimizer_params = optimizer_params.get('generator', {})
d_optimizer_params = optimizer_params.get('discriminator', {})
trainer_params = cfg_dict.get('trainer', {})
inferencer_params = cfg_dict.get('inferencer', {})
stats_params = cfg_dict.get('stats', {})

logger.info('Building Models ...')
model_name = model_params.get('name', 'DCGAN')
latent_dim = model_params.get('latent_dim', 100)

if model_name == 'DCGAN':
    G = DCGAN_Generator(**g_model_params)
    D = DCGAN_Discriminator(**d_model_params)
else:
    raise NotImplementedError('Invalid model name')

multigpu = trainer_params.get('multigpu', False)
if multigpu:
    logger.info('Initialize multi-gpu training ...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cpu'):
        raise EnvironmentError('No GPUs, cannot initialize multigpu training.')
    G.to(device)
    D.to(device)
else:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    G.to(device)
    D.to(device)

dataset_type = dataset_params.get('type', 'CelebA')
if dataset_type == 'CelebA':
    logger.info('Building {} datasets ...'.format(dataset_type))
    dataset = CelebADataset(
        root = dataset_params.get('path', 'data'),
        split = 'whole',
        img_size = dataset_params.get('img_size', 64),
        center_crop = dataset_params.get('center_crop', 148)
    )
elif dataset_type == 'CUB200':
    logger.info('Building {} datasets ...'.format(dataset_type))
    dataset = Cub200Dataset(
        root = dataset_params.get('path', 'data'),
        split = 'whole',
        img_size = dataset_params.get('img_size', 64),
        center_crop_scale = dataset_params.get('center_crop_scale', 1.2)
    )
else:
    raise NotImplementedError('Invalid dataset type.')

logger.info('Building dataloader ...')
batch_size = dataset_params.get('batch_size', 64)
dataloader = DataLoader(
    dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 4,
    drop_last = True
)

logger.info('Building optimizer and learning rate scheduler ...')
g_optimizer = torch.optim.AdamW(
    G.parameters(),
    betas = (g_optimizer_params.get('adam_beta1', 0.5), g_optimizer_params.get('adam_beta2', 0.999)),
    lr = g_optimizer_params.get('learning_rate', 0.0002),
    weight_decay = g_optimizer_params.get('weight_decay', 0.01),
    eps = g_optimizer_params.get('eps', 1e-6)
)
d_optimizer = torch.optim.AdamW(
    D.parameters(),
    betas = (d_optimizer_params.get('adam_beta1', 0.5), d_optimizer_params.get('adam_beta2', 0.999)),
    lr = d_optimizer_params.get('learning_rate', 0.0002),
    weight_decay = d_optimizer_params.get('weight_decay', 0.01),
    eps = d_optimizer_params.get('eps', 1e-6)
)

criterion = nn.BCELoss()

logger.info('Checking checkpoints ...')
start_epoch = 0
max_epoch = trainer_params.get('max_epoch', 200)
stats_dir = os.path.join(stats_params.get('stats_dir', 'stats'), stats_params.get('stats_folder', 'temp'))
if os.path.exists(stats_dir) == False:
    os.makedirs(stats_dir)
checkpoint_file = os.path.join(stats_dir, 'checkpoint.tar')
if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    G.load_state_dict(checkpoint['g_model_state_dict'])
    D.load_state_dict(checkpoint['d_model_state_dict'])
    start_epoch = checkpoint['epoch']
    logger.info("Checkpoint {} (epoch {}) loaded.".format(checkpoint_file, start_epoch))
elif MODE == "inference":
    raise AttributeError('There should be a checkpoint file for inference.')

multigpu = trainer_params.get('multigpu', False)
if multigpu:
    G = torch.nn.DataParallel(G)
    D = torch.nn.DataParallel(D)


def combined_train_one_epoch(epoch):
    logger.info('Start combined training process in epoch {}.'.format(epoch + 1))
    G.train()
    D.train()
    g_losses = []
    d_losses = []
    with tqdm(dataloader) as pbar:
        for data in pbar:
            if CLEAN_CACHE and device != torch.device('cpu'):
                torch.cuda.empty_cache()
            imgs, img_labels = data
            imgs = imgs.to(device)
            img_labels = img_labels.to(device)
            cur_batch_size = imgs.shape[0]

            # Train D with real images and fake images
            d_optimizer.zero_grad()
            labels = torch.ones(cur_batch_size, dtype = torch.float32, device = device)
            res_real = D(imgs, label = img_labels)
            loss_real = criterion(res_real, labels)

            noise = torch.randn(cur_batch_size, latent_dim, dtype = torch.float32, device = device)
            fake_imgs = G(noise, label = img_labels)
            labels = torch.zeros(cur_batch_size, dtype = torch.float32, device = device)
            res_fake = D(fake_imgs, label = img_labels)
            loss_fake = criterion(res_fake, labels)

            loss = loss_real + loss_fake
            loss.backward(retain_graph = True)
            d_optimizer.step()

            # Train G with D
            g_optimizer.zero_grad()
            labels = torch.ones(cur_batch_size, dtype = torch.float32, device = device)
            res_fake4real = D(fake_imgs, label = img_labels)
            loss_fake4real = criterion(res_fake4real, labels)
            loss_fake4real.backward()
            g_optimizer.step()

            # Output logs
            D_x = res_real.mean().item()
            D_G_z1 = res_fake.mean().item()
            D_G_z2 = res_fake4real.mean().item()
            
            pbar.set_description('Epoch {}, D loss: {:.4f}, G loss: {:.4f}, D(x): {:.4f}, D(G(z)): {:.4f} / {:.4f}, '.format(
                epoch + 1, loss.item(), loss_fake4real.item(),
                D_x, D_G_z1, D_G_z2,
            ))
            g_losses.append(loss_fake4real.item())
            d_losses.append(loss.item())
    mean_g_loss = np.array(g_losses).mean()
    mean_d_loss = np.array(d_losses).mean()
    logger.info('Finish training process in epoch {}, mean G loss: {:.8f}, mean D loss: {:.8f}'.format(epoch + 1, mean_g_loss, mean_d_loss))
    return mean_g_loss, mean_d_loss


def inference(epoch = -1):
    suffix = ""
    if 0 <= epoch < max_epoch:
        logger.info('Begin inference on checkpoint of epoch {} ...'.format(epoch + 1))
        suffix = "epoch_{}".format(epoch)
    else:
        logger.info('Begin inference ...')
    sample_num = inferencer_params.get('sample_num', batch_size)
    if sample_num > batch_size:
        raise AttributeError('Sample number should be less than batch size.')
    _, img_labels = next(iter(dataloader))
    img_labels = img_labels[:sample_num]
    img_labels = img_labels.to(device)
    noise = torch.randn(sample_num, latent_dim, dtype = torch.float32, device = device)
    with torch.no_grad():
        img = G(noise, label = img_labels)
    img = img.detach().cpu()
    nrow = int(np.ceil(np.sqrt(sample_num)))
    generated_dir = os.path.join(stats_dir, 'generated_images')
    if os.path.exists(generated_dir) == False:
        os.makedirs(generated_dir)
    tuitls.save_image(
        img.data,
        os.path.join(generated_dir, "generated_{}.png".format(suffix)),
        normalize = True,
        nrow = nrow
    )
    logger.info('Finish inference successfully.')


def train(start_epoch):
    global cur_epoch
    for epoch in range(start_epoch, max_epoch):
        cur_epoch = epoch
        logger.info('--> Begin training epoch {}/{}'.format(epoch + 1, max_epoch))
        mean_g_loss, mean_d_loss = combined_train_one_epoch(epoch)
        if multigpu is False:
            save_dict = {
                'epoch': epoch + 1,
                'g_loss': mean_g_loss,
                'd_loss': mean_d_loss,
                'g_model_state_dict': G.state_dict(),
                'd_model_state_dict': D.state_dict()
            }
        else:
            save_dict = {
                'epoch': epoch + 1, 
                'g_loss': mean_g_loss,
                'd_loss': mean_d_loss,
                'g_model_state_dict': G.module.state_dict(),
                'd_model_state_dict': D.module.state_dict()
            }
        torch.save(save_dict, os.path.join(stats_dir, 'checkpoint.tar'))
        inference(epoch)


if __name__ == '__main__':
    if MODE == "train":
        train(start_epoch)
    else:
        inference()