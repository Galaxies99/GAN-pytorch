from models.WGAN import WGAN_Discriminator, WGAN_Generator
import torch
import torch.nn as nn
import torch.nn.functional as F


def optimizer_builder(model, optimizer_params):
    from torch.optim import SGD, ASGD, Adagrad, Adamax, Adadelta, Adam, AdamW, RMSprop
    type = optimizer_params.get('type', 'Adam')
    params = optimizer_params.get('params', {})
    if type == 'SGD':
        optimizer = SGD(model.parameters(), **params)
    elif type == 'ASGD':
        optimizer = ASGD(model.parameters(), **params)
    elif type == 'Adagrad':
        optimizer = Adagrad(model.parameters(), **params)
    elif type == 'Adamax':
        optimizer = Adamax(model.parameters(), **params)
    elif type == 'Adadelta':
        optimizer = Adadelta(model.parameters(), **params)
    elif type == 'Adam':
        optimizer = Adam(model.parameters(), **params)
    elif type == 'AdamW':
        optimizer = AdamW(model.parameters(), **params)
    elif type == 'RMSprop':
        optimizer = RMSprop(model.parameters(), **params)
    else:
        raise NotImplementedError('Invalid optimizer type.')
    return optimizer


def model_builder(model_params):
    from models.DCGAN import DCGAN_Generator, DCGAN_Discriminator
    model_name = model_params.get('name', 'DCGAN')
    latent_dim = model_params.get('latent_dim', 100)
    g_model_params = model_params.get('generator', {})
    d_model_params = model_params.get('discriminator', {})
    g_model_params['latent_dim'] = latent_dim
    if model_name == 'DCGAN':
        G = DCGAN_Generator(**g_model_params)
        D = DCGAN_Discriminator(**d_model_params)
    elif model_name == 'WGAN':
        G = WGAN_Generator(**g_model_params)
        D = WGAN_Discriminator(**d_model_params)
    else:
        raise NotImplementedError('Invalid model type.')
    return G, D


def dataset_builder(dataset_params):
    from datasets.celeba import CelebADataset
    from datasets.cub200 import Cub200Dataset
    dataset_type = dataset_params.get('type', 'CelebA')
    if dataset_type == 'CelebA':
        dataset = CelebADataset(
            root = dataset_params.get('path', 'data'),
            split = 'whole',
            img_size = dataset_params.get('img_size', 64),
            center_crop = dataset_params.get('center_crop', 148)
        )
    elif dataset_type == 'CUB200':
        dataset = Cub200Dataset(
            root = dataset_params.get('path', 'data'),
            split = 'whole',
            img_size = dataset_params.get('img_size', 64),
            center_crop_scale = dataset_params.get('center_crop_scale', 1.2)
        )
    else:
        raise NotImplementedError('Invalid dataset type.')
    return dataset
