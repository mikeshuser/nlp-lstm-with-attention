#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import json
from random import shuffle
from math import ceil
from typing import List, Tuple, Union, Iterator
from collections import Counter
import tensorflow as tf
from numpy import array, eye

# object type returned from the DataManager
DataSourceType = Iterator[
    Tuple[
        List[List[int]], array # (sequence tokens, 1-hot label array)
    ]
]

class DataManager():
    
    """
    Custom generator class to yield X, y batches.
    Also handles the tokenization.
    Document padding is 'post'.

    Returns train and validation(optional) data
    
    Expects the data as a list of (sample, label) tuples

    kwargs provided through config file:
    batch_size
    val_percent
    max_len
    TODO random_seed
    """

    def __init__(self, 
        train_data: List[Tuple[str, int]],
        test_data: List[Tuple[str, int]],
        config: Union[str, dict], 
        **kwargs
    ):

        if isinstance(config, str):
            with open(config, "r") as path:
                config = json.load(path)['data']
        elif isinstance(config, dict):
            config = config['data']
        else:
            raise TypeError("config needs to be type str or dict")


        self.labels = config.get('labels')
        self.num_classes = len(self.labels)
        self.batch_size = config.get('batch_size')
        self.val_percent = config.get('val_percent')
        self.max_len = config.get('max_len')
        self.pad_value = config.get('pad_value', 0)
        self.vocab = {}
        self.inverse_vocab = {}

        self.train_data, self.val_data = self._split_train(train_data)
        self.test_data = test_data

        self.end_of_epoch()

    def _split_train(self, train_data: List[Tuple[str, int]]):
        
        """
        Pull validation samples out of the training dataset
        """

        all_indices = list(range(len(train_data)))
        shuffle(all_indices)

        partitions = {k : [
            train_data[i] for i in all_indices if int(train_data[i][1]) == k
            ] for k in range(1, self.num_classes + 1)
        }

        final_train = []
        final_val = []
        for k in range(1, self.num_classes + 1):
            threshold = ceil(len(partitions[k]) * self.val_percent)
            final_val.extend(partitions[k][:threshold])
            final_train.extend(partitions[k][threshold:])

        return final_train, final_val

    def build_vocab(self, drop_less_than: int = 2):

        """
        Build vocab dictionary from self.train_data
        """

        c = Counter()
        for doc in self.train_data:
            c.update(doc[0].split())

        self.vocab['<pad>'] = self.pad_value
        self.vocab['<unk>'] = 1
        idx = 2
        for token, count in c.items():
            if count >= drop_less_than: 
                self.vocab[token] = idx
                idx += 1

        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def batched_dataset(self, dataset_type: str) -> DataSourceType:
        
        """
        Returns a data generator to step through the dataset in batches

        dataset_type: should be one of ('train', 'val', 'test')
        """
        
        assert dataset_type in ('train', 'val', 'test')
        if dataset_type == 'train':
            data = self.train_data
        elif dataset_type == 'val':
            data = self.val_data
        elif dataset_type == 'test':
            data = self.test_data

        steps = ceil(len(data) / self.batch_size)
        i = 0
        while i < steps:
            batch = data[i * self.batch_size : (i+1) * self.batch_size]
            X = []
            y = []

            for row in batch:
                doc = row[0].split()
                tmp = []
                for j in range(self.max_len): 
                    try:
                        tmp.append(self.str2idx(doc[j]))
                    except IndexError:
                        tmp.append(self.str2idx('<pad>'))

                X.append(tmp)
                y.append(int(row[1]) - 1) # 0-based index for 1-hot encoding

            yield X, eye(self.num_classes)[y]
            i += 1

    def end_of_epoch(self):
        shuffle(self.train_data)
        shuffle(self.val_data)

    def str2idx(self, tok: str):
        return self.vocab.get(tok, self.vocab['<unk>'])

    def idx2str(self, idx: int):
        return self.inverse_vocab.get(idx, '<unk>')
