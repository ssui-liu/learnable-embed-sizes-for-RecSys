import math
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader.movielens import MovieLensDataset
from data_loader.avazu import AvazuDataset
from data_loader.criteo import CriteoDataset


def setup_generator(opt):
    """Choose different type of sampler for MF & FM"""
    if opt['factorizer'] == 'fm':
        return FMGenerator(opt)
    else:
        raise NotImplementedError


class FMGenerator(object):
    def __init__(self, opt):
        data_path = opt['data_path']
        data_type = opt['data_type']
        category_only = opt['category_only']
        rebuild_cache = opt['rebuild_cache']
        self.batch_size_train = opt.get('batch_size_train')
        self.batch_size_valid = opt.get('batch_size_valid')
        self.batch_size_test = opt.get('batch_size_test')

        self.opt = opt

        if data_type == 'criteo':
            dataset = CriteoDataset(data_path+'train.txt', data_path+'cache', rebuild_cache=rebuild_cache)
        elif data_type == 'avazu':
            dataset = AvazuDataset(data_path+'train', data_path+'cache', rebuild_cache=rebuild_cache)
        elif data_type == 'ml-1m':
            dataset = MovieLensDataset(data_path, data_type)
        else:
            raise RuntimeError("Invalid data type: {}".format(data_type))

        train_length = int(len(dataset) * 0.8)
        valid_length = int(len(dataset) * 0.1)
        test_length = len(dataset) - train_length - valid_length
        self.train_data, self.valid_data, self.test_data = torch.utils.data.random_split(
                dataset, (train_length, valid_length, test_length))

        self._train_epoch = iter([])
        self._valid_epoch = iter([])
        self._test_epoch = iter([])

        self.num_batches_train = math.ceil(len(self.train_data) / self.batch_size_train)
        self.num_batches_valid = math.ceil(len(self.valid_data) / self.batch_size_valid)
        self.num_batches_test = math.ceil(len(self.test_data) / self.batch_size_test)
        if data_type == 'criteo' and category_only:
            self.field_dims = dataset.field_dims[13:]
        else:
            self.field_dims = dataset.field_dims

        print('\tNum of train records: {}'.format(len(self.train_data)))
        print('\tNum of valid records: {}'.format(len(self.valid_data)))
        print('\tNum of test records: {}'.format(len(self.test_data)))
        print('\tNum of fields: {}'.format(len(self.field_dims)))
        print('\tNum of features: {}'.format(sum(self.field_dims)))

    @property
    def train_epoch(self):
        """list of training batches"""
        return self._train_epoch

    @train_epoch.setter
    def train_epoch(self, new_epoch):
        self._train_epoch = new_epoch

    @property
    def valid_epoch(self):
        """list of validation batches"""
        return self._valid_epoch

    @valid_epoch.setter
    def valid_epoch(self, new_epoch):
        self._valid_epoch = new_epoch

    @property
    def test_epoch(self):
        """list of test batches"""
        return self._test_epoch

    @test_epoch.setter
    def test_epoch(self, new_epoch):
        self._test_epoch = new_epoch

    def get_epoch(self, type):
        """
        return:
            list, an epoch of batchified samples of type=['train', 'valid', 'test']
        """
        if type == 'train':
            return self.train_epoch

        if type == 'valid':
            return self.valid_epoch

        if type == 'test':
            return self.test_epoch

    def get_sample(self, type):
        """get training sample or validation sample"""
        epoch = self.get_epoch(type)

        try:
            sample = next(epoch)
        except StopIteration:
            self.set_epoch(type)
            epoch = self.get_epoch(type)
            sample = next(epoch)
            if self.opt['load_in_queue']:
                # continue to queue
                self.cont_queue(type)

        return sample

    def set_epoch(self, type):
        """setup batches of type = [training, validation, testing]"""
        # print('\tSetting epoch {}'.format(type))
        start = datetime.now()
        if type == 'train':
            loader = DataLoader(self.train_data,
                                batch_size=self.batch_size_train,
                                shuffle=True, pin_memory=False)
            self.train_epoch = iter(loader)
            num_batches = len(self.train_epoch)
        elif type == 'valid':

            loader = DataLoader(self.valid_data,
                                batch_size=self.batch_size_valid,
                                shuffle=True, pin_memory=False)
            self.valid_epoch = iter(loader)
            num_batches = len(self.valid_epoch)
        elif type == 'test':

            loader = DataLoader(self.test_data,
                                batch_size=self.batch_size_test,
                                shuffle=False, pin_memory=False)
            self.test_epoch = iter(loader)
            num_batches = len(self.test_epoch)
        end = datetime.now()
        # print('\tFinish setting epoch {}, num_batches {}, time {} mins'.format(type,
        #                                                                        num_batches,
        #                                                                        (end - start).total_seconds() / 60))



