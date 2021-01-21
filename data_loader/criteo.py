import math
import shutil
import struct
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import lmdb
import numpy as np
import torch.utils.data
from tqdm import tqdm
import pandas as pd


class CriteoDataset(torch.utils.data.Dataset):
    """
    Criteo Display Advertising Challenge Dataset

    Data prepration:
        * Remove the infrequent features (appearing in less than threshold instances) and treat them as a single feature
        * Discretize numerical values by log2 transformation which is proposed by the winner of Criteo Competition

    :param dataset_path: criteo train.txt path.
    :param cache_path: lmdb cache path.
    :param rebuild_cache: If True, lmdb cache is refreshed.
    :param min_threshold: infrequent feature threshold.

    Reference:
        https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset
        https://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf
    """

    def __init__(self, dataset_path=None, cache_path='.criteo', rebuild_cache=False, min_threshold=10,
                 category_only=False):  # category
        self.NUM_FEATS = 39
        self.NUM_INT_FEATS = 13
        self.min_threshold = min_threshold
        self.category_only = category_only
        self.item_idx = 0
        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError('create cache: failed: dataset_path is None')
            self.__build_cache(dataset_path, cache_path)
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        print(self.item_idx)
        with self.env.begin(write=False) as txn:
            stat = txn.stat()
            self.length = stat['entries'] - 1
            self.field_dims = np.frombuffer(txn.get(b'field_dims'), dtype=np.uint32)

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            name = struct.pack('>I', index)
            stream = txn.get(name)
            if stream is None:
                print("None")
                print(index)
            np_array = np.frombuffer(stream, dtype=np.uint32).astype(dtype=np.long)
        if self.category_only:
            return np_array[1 + self.NUM_INT_FEATS:], np_array[0]
        else:
            return np_array[1:], np_array[0]

    def __len__(self):
        return self.length

    def __build_cache(self, path, cache_path):
        feat_mapper, defaults, field_dims = self.__get_feat_mapper(path)
        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            with env.begin(write=True) as txn:
                txn.put(b'field_dims', field_dims.tobytes())
            for buffer in self.__yield_buffer(path, feat_mapper, defaults):
                with env.begin(write=True) as txn:
                    for key, value in buffer:
                        txn.put(key, value)

    def __get_feat_mapper(self, path):
        feat_cnts = defaultdict(lambda: defaultdict(int))
        new_feat_cnts = defaultdict(lambda: defaultdict(int))
        with open(path) as f:
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create criteo dataset cache: counting features')
            for line in pbar:
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                for i in range(1, self.NUM_INT_FEATS + 1):
                    feat_cnts[i][convert_numeric_feature(values[i])] += 1
                for i in range(self.NUM_INT_FEATS + 1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1

        feat_mapper = {i: {feat for feat, c in cnt.items() if c >= self.min_threshold} for i, cnt in feat_cnts.items()}
        feat_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)} for i, cnt in feat_mapper.items()}
        defaults = {i: len(cnt) for i, cnt in feat_mapper.items()}

        for field, sub_dict in feat_cnts.items():
            for key in list(sub_dict.keys()):
                if sub_dict[key] < self.min_threshold:
                    sub_dict['default'] += 1
                else:
                    new_feat_cnts[field][feat_mapper[field][key]] = sub_dict[key]
            if sub_dict['default'] != 0:
                new_feat_cnts[field][len(feat_mapper[field])] = sub_dict['default']
        field_dims = self.__get_field_dims(new_feat_cnts)
        return feat_mapper, defaults, field_dims

    def __yield_buffer(self, path, feat_mapper, defaults, buffer_size=int(1e5)):
        item_idx = 0
        buffer = list()
        with open(path) as f:
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create criteo dataset cache: setup lmdb')
            for line in pbar:
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                np_array = np.zeros(self.NUM_FEATS + 1, dtype=np.uint32)
                np_array[0] = int(values[0])
                for i in range(1, self.NUM_INT_FEATS + 1):
                    np_array[i] = feat_mapper[i].get(convert_numeric_feature(values[i]), defaults[i])
                for i in range(self.NUM_INT_FEATS + 1, self.NUM_FEATS + 1):
                    np_array[i] = feat_mapper[i].get(values[i], defaults[i])
                name = struct.pack('>I', item_idx)
                if name is None:
                    print("None")
                buffer.append((name, np_array.tobytes()))
                item_idx += 1
                if item_idx % buffer_size == 0:
                    yield buffer
                    buffer.clear()

            self.item_idx = item_idx
            yield buffer

    def __get_field_dims(self, data):
        all_freq = None
        index_offset = 0
        field_dims = np.zeros(self.NUM_FEATS, dtype=np.uint32)
        for i, col in enumerate(data.keys()):
            freq = pd.Series(data[col]).sort_values(ascending=False)
            freq.index = freq.index + index_offset
            if all_freq is None:
                all_freq = freq
            else:
                all_freq = pd.concat([all_freq, freq], axis=0)
            index_offset += len(freq)
            field_dims[i] = len(freq)

        return field_dims


@lru_cache(maxsize=None)
def convert_numeric_feature(val: str):
    if val == '':
        return 'NULL'
    v = int(val)
    if v > 2:
        return str(int(math.log(v) ** 2))
    else:
        return str(v - 2)
