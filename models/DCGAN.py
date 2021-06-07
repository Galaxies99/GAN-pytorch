import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class DCGAN_Generator(nn.Module):
    def __init__(self, channels, latent_dim, hidden_dim, **kwargs):
        super(DCGAN_Generator, self).__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.epoch = kwargs.get('epoch', 0)
        self.model = nn.Sequential(
            Rearrange('b d -> b d 1 1'),
            self._make_layers(self.latent_dim, self.hidden_dim * 8, stride = 1, padding = 0),
            self._make_layers(self.hidden_dim * 8, self.hidden_dim * 4),
            self._make_layers(self.hidden_dim * 4, self.hidden_dim * 2),
            self._make_layers(self.hidden_dim * 2, self.hidden_dim),
            self._make_layers(self.hidden_dim, self.channels, with_bn = False, activation = 'Tanh')
        )
        self.loss_function = nn.BCELoss()
    
    def _make_layers(self, in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, output_padding = 0, bias = False, with_bn = True, activation = 'ReLU'):
        if activation not in ['ReLU', 'Tanh']:
            raise NotImplementedError('Activation function of generator of DCGAN should be either "ReLU" or "Tanh".')
        layers = [nn.ConvTranspose2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            output_padding = output_padding,
            bias = bias
        )]
        if with_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace = True) if activation == 'ReLU' else nn.Tanh())
        return nn.Sequential(*layers)
    
    def forward(self, z, **kwargs):
        return self.model(z)
    
    def loss(self, z, discriminator, **kwargs):
        res = self.forward(z)
        cur_batch_size = z.shape[0]
        labels = torch.ones(cur_batch_size, dtype = torch.float32, device = z.device)
        res_fake4real = discriminator(res)
        return self.loss_function(res_fake4real, labels)

    def annealing_update(self, *args, **kwargs):
        self.epoch = self.epoch + 1


class DCGAN_Discriminator(nn.Module):
    def __init__(self, channels, hidden_dim, **kwargs):
        super(DCGAN_Discriminator, self).__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.epoch = kwargs.get('epoch', 0)
        noise_adding_params = kwargs.get('noise_adding', {})
        self.noise_adding = noise_adding_params.get('activation', False)
        self.noise_initial_sigma = noise_adding_params.get('initial_sigma', 0.5)
        self.noise_annealing_epochs = noise_adding_params.get('annealing_epochs', 0)
        self.model = nn.Sequential(
            self._make_layers(self.channels, self.hidden_dim, with_bn = False),
            self._make_layers(self.hidden_dim, self.hidden_dim * 2),
            self._make_layers(self.hidden_dim * 2, self.hidden_dim * 4),
            self._make_layers(self.hidden_dim * 4, self.hidden_dim * 8),
            self._make_layers(self.hidden_dim * 8, 1, stride = 1, padding = 0, with_bn = False, activation = 'Sigmoid'),
            Rearrange('b 1 1 1 -> b')
        )
        self.loss_function = nn.BCELoss()

    def _make_layers(self, in_channels, out_channels, kernel_size = 4, stride = 2, padding = 1, bias = False, with_bn = True, activation = 'LeakyReLU'):
        if activation not in ['LeakyReLU', 'Sigmoid']:
            raise NotImplementedError('Activation function of discriminator of DCGAN should be either "LeakyReLU" or "Sigmoid".')
        layers = [nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            bias = bias
        )]
        if with_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace = True) if activation == 'LeakyReLU' else nn.Sigmoid())
        return nn.Sequential(*layers)
    
    def forward(self, x, **kwargs):
        if self.noise_adding and (self.noise_annealing_epochs == -1 or self.epoch < self.noise_annealing_epochs):
            if self.noise_annealing_epochs == -1:
                sigma = self.noise_initial_sigma
            else:
                sigma = self.noise_initial_sigma / self.noise_annealing_epochs * (self.noise_annealing_epochs - self.epoch)
            noise = torch.randn_like(x) * sigma
            x = x + noise
        return self.model(x)
    
    def loss(self, img, realness, **kwargs):
        res = self.forward(img)
        cur_batch_size = img.shape[0]
        if realness:
            labels = torch.ones(cur_batch_size, dtype = torch.float32, device = img.device)
        else:
            labels = torch.zeros(cur_batch_size, dtype = torch.float32, device = img.device)
        return self.loss_function(res, labels)
    
    def annealing_update(self, *args, **kwargs):
        self.epoch = self.epoch + 1
