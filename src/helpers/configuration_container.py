import logging
import os

import sys

from helpers.singleton import Singleton
import importlib


@Singleton
class ConfigurationContainer:
    class_maps = {
        'bceloss': ('torch.nn', 'BCELoss'),
        'mnist': ('data.mnist_data_loader', 'MNISTDataLoader'),
        'cifar': ('data.cifar10_data_loader', 'CIFAR10DataLoader'),
        'celeba': ('data.celeba_data_loader', 'CelebADataLoader'),
        'four_layer_perceptron': ('networks.network_factory', 'FourLayerPerceptronFactory'),
        'convolutional': ('networks.network_factory', 'ConvolutionalNetworkFactory'),
        'seabiscuit_gan': ('training.ea.seabiscuit_gan_trainer', 'SeabiscuitGANTrainer'),
        'seabiscuit_wgan': ('training.ea.seabiscuit_wgan_trainer', 'SeabiscuitWGANTrainer'),
    }

    _logger = logging.getLogger(__name__)

    def __init__(self):
        self.settings = {}
        self._output_dir = None

    def create_instance(self, name, *args):
        module_name, class_name = self.class_maps[name]
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls(*args)

    # Encapsulated properties for often used settings
    @property
    def is_losswise_enabled(self):
        """
        :return: true if losswise sections exist and status is set to enabled
        """
        return 'losswise' in self.settings['general'] \
               and 'enabled' in self.settings['general']['losswise'] \
               and self.settings['general']['losswise']['enabled']

    @property
    def output_dir(self):
        """
        Also creates the output directory if it does not yet exists.
        :return: The output directoy specified in config file, combined with a method-specific subfolder
        """

        if self._output_dir is None:
            self._output_dir = self._load_output_dir()
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        self._output_dir = value

    def _load_output_dir(self):
        output = self.settings['general']['output_dir'] if 'output_dir' in self.settings['general'] else 'output'
        subdir = self.settings['trainer']['method']['name'] if 'method' in self.settings['trainer'] else \
            self.settings['trainer']['name']
        directory = os.path.join(output, subdir)

        if not os.path.exists(directory):
            os.makedirs(directory)

        return directory
