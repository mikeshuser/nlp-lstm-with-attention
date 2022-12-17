#!/usr/bin/env python3
# -*- coding: utf-8 -*- 

import os
import time
import json
import random
from itertools import product
from typing import Iterator, List, Tuple
import pandas as pd
import numpy as np
import tensorflow as tf
from models import (
    TextRNN, 
    DataManager,
    DataSourceType, 
    text_pipe,
)
from main import (
    train_loop,
    train_step,
    val_step,
    test_step,
    evaluate,
)

NUM_EXPERIMENTS = 30
RANDOM_SEED = 42
EPOCHS = 10

def dict_walk(d: dict, res: dict):
    
    """
    recursive function to walk through dictionary  
    """

    for k, v in d.items():
        if isinstance(v, dict): 
            dict_walk(v, res)
        elif isinstance(v, list):
            res[k] = v

def build_grid(search_space):

    """
    convert hyper parameter values from config into grid search space
    """

    hyper_params = {} 
    dict_walk(search_space, hyper_params)
    param_values = list(hyper_params.values())
    grid = list(product(*param_values))
    
    return list(hyper_params.keys()), grid

def init_model_config(variable_params, scenario):

    """
    intilialize model config for a given scenario
    """
    params = dict(zip(variable_params, scenario))
    model_config = {
        'dimensions' : {},
        'optimizer' : {},
        'regularizer' : {},
    }

    model_config['dimensions']['embedding_dim'] = params['embedding_dim']
    model_config['dimensions']['recurrent_units'] = params['recurrent_units']
    model_config['dimensions']['linear_units'] = params['linear_units']

    model_config['optimizer']['type'] = 'adam'
    model_config['optimizer']['init_lr'] = params['init_lr']

    model_config['regularizer']['dropout_rate'] = params['dropout_rate']
    model_config['regularizer']['weights'] = {
        'type' : 'L2',
        'lambda' : params['lambda']
    }

    return model_config

if __name__ == "__main__":


    with open("./experimental_config.json", "r") as path:
        config = json.load(path)
    
    variable_params, search_grid = build_grid(config['search_space'])

    random.seed(RANDOM_SEED)
    random.shuffle(search_grid)
    scenarios = search_grid[:NUM_EXPERIMENTS]

    train_docs = text_pipe("data/ag_news_csv/train.csv", 2, 0)
    test_docs = text_pipe("data/ag_news_csv/test.csv", 2, 0)
    dm = DataManager(train_docs, test_docs, config)
    dm.build_vocab(drop_less_than=config['data']['vocab_drop_threshold'])
    res_df = pd.DataFrame(columns=[
        "id",
        *variable_params,
        "best_val_epoch",
        "val_loss",
        "val_accuracy",
        "val_auc",
        "test_loss",
        "test_accuracy",
        "test_auc",
    ])

    # main experiment loop
    for i, scenario in enumerate(scenarios):

        scenario_config = init_model_config(variable_params, scenario)
        print(f"running scenario {i + 1}")
        print(scenario_config)
        scenario_config['data'] = config['data']
        model_id = f"scenario{i + 1}"
        res_df.at[i, 'id'] = model_id
        #save scenario nums

        # optimizer, loss, metrics
        if config['optimizer']['type'] == 'adam':
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=config['optimizer']['init_lr'], 
                clipnorm=1.
            )
        else:
            # TODO test for other optimizers
            pass

        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

        metrics = dict(
            train=dict(
                acc=tf.keras.metrics.CategoricalAccuracy(name='acc'),
                auc=tf.keras.metrics.AUC(curve='ROC', name='roc-auc'),
            ),
            val=dict(
                acc=tf.keras.metrics.CategoricalAccuracy(name='acc'),
                auc=tf.keras.metrics.AUC(curve='ROC', name='roc-auc'),
            ),
            test=dict(
                acc=tf.keras.metrics.CategoricalAccuracy(name='acc'),
                auc=tf.keras.metrics.AUC(curve='ROC', name='roc-auc'),
            ),
        )

        train_writer = tf.summary.create_file_writer(f'./logs/train{i + 1}/')
        val_writer  = tf.summary.create_file_writer(f'./logs/val{i + 1}/')

        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)

        nn = TextRNN(scenario_config, len(dm.vocab))
        nn.build_graph()
        curves = train_loop(dm, model_id, EPOCHS, 1, True)

        # find epoch with best val loss
        best_epoch = np.argmin(curves['loss'])
        res_df.at[i, 'best_val_epoch'] = best_epoch + 1
        res_df.at[i, 'val_loss'] = curves['loss'][best_epoch]
        res_df.at[i, 'val_accuracy'] = curves['acc'][best_epoch]
        res_df.at[i, 'val_auc'] = curves['auc'][best_epoch]

        # load scenario best model
        best_nn = TextRNN(scenario_config, len(dm.vocab))
        best_nn.build((
            None,
            scenario_config['data']['max_len'], 
            scenario_config['dimensions']['embedding_dim']
        ))
        best_nn.load_weights(f"./checkpoints/{model_id}_epoch{best_epoch}.h5")

        # eval best model on test data
        test_loss = evaluate(dm)
        res_df.at[i, 'test_loss'] = test_loss
        res_df.at[i, 'test_accuracy'] = metrics['test']['acc']
        res_df.at[i, 'test_auc'] = metrics['test']['auc']      

        # delete scenario checkpoints
        _ = [os.remove(
                os.path.join("./checkpoints", f)
            ) for f in os.listdir("./checkpoints")
        ]

    res_df.to_excel("./experiment_results.xlsx", index=False)