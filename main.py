#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import json
from typing import Iterator, List, Tuple
import tensorflow as tf
from models import (
    TextRNN, 
    DataManager,
    DataSourceType, 
    text_pipe,
)

DIAGNOSTIC = """Epoch: {}, time taken: {}
    train - loss: {:4.4f}, Accuracy {:4.4f}, AUC {:4.4f}
    val  -  loss: {:4.4f}, Accuracy {:4.4f}, AUC {:4.4f}"""

def train_loop(
    dm: DataManager,
    model_id: str,
    epochs: int,
    checkpoint_every: int = 0,
    experiment_mode: bool = False,
):

    """
    Main model training loop
    """

    train_data = dm.batched_dataset('train')
    val_data = dm.batched_dataset('val')

    for epoch in range(1, epochs + 1):
        t = time.time()

        # training phase
        for step, (batch, labels) in enumerate(dm.batched_dataset('train')):

            step = tf.constant(step)
            x_train = tf.convert_to_tensor(batch, dtype=tf.int32)
            y_train = tf.convert_to_tensor(labels, dtype=tf.int32)

            train_loss = train_step(step, x_train, y_train)

        # validation phase
        for step, (batch, labels) in enumerate(dm.batched_dataset('val')):

            step = tf.constant(step)
            x_val = tf.convert_to_tensor(batch, dtype=tf.int32)
            y_val = tf.convert_to_tensor(labels, dtype=tf.int32)

            val_loss = val_step(step, x_val, y_val)

        # metrics
        with train_writer.as_default():
            tf.summary.scalar('loss', train_loss, step=epoch)
            tf.summary.scalar(
                'accuracy', 
                metrics['train']['acc'].result(), 
                step=epoch
            ) 
            tf.summary.scalar(
                'roc-auc', 
                metrics['train']['auc'].result(), 
                step=epoch
            ) 
            train_writer.flush()

        with val_writer.as_default():
            tf.summary.scalar('loss', val_loss, step=epoch)
            tf.summary.scalar(
                'accuracy', 
                metrics['val']['acc'].result(), 
                step=epoch
            ) 
            tf.summary.scalar(
                'roc-auc', 
                metrics['val']['auc'].result(), 
                step=epoch
            ) 
            val_writer.flush()

        print(DIAGNOSTIC.format(
            epoch, 
            round((time.time() - t)/60, 2),
            train_loss, 
            metrics['train']['acc'].result(), 
            metrics['train']['auc'].result(),
            val_loss, 
            metrics['val']['acc'].result(), 
            metrics['val']['auc'].result()
        ))

        for phase in ['train', 'val']:
            for metric in metrics[phase].keys():
                metrics[phase][metric].reset_state()
        dm.end_of_epoch()
        
        if checkpoint_every != 0:
            if (epoch) % checkpoint_every == 0:
                print("saving checkpoint...")
                nn.save_weights(f"./checkpoints/{model_id}_epoch{epoch}.h5")

    if experiment_mode:
        return # stats
    else:
        return nn

@tf.function
def train_step(step, x, y):
    with tf.GradientTape() as tape:
        train_output, attention = nn(x, training=True)
        train_loss_value = loss_fn(y, train_output)
        train_loss_value += sum(nn.losses)
        
    grads = tape.gradient(train_loss_value, nn.trainable_weights)
    optimizer.apply_gradients(zip(grads, nn.trainable_weights))
    
    for metric in metrics['train'].keys():
        metrics['train'][metric].update_state(y, train_output)

    return train_loss_value

@tf.function
def val_step(step, x, y):
    val_output, attention = nn(x, training=False)
    val_loss_value = loss_fn(y, val_output)
    
    for metric in metrics['val'].keys():
        metrics['val'][metric].update_state(y, val_output)
        
    return val_loss_value


if __name__ == "__main__":

    with open("./main_config.json", "r") as path:
        config = json.load(path)
    
    model_id = "lstm1"

    # data prep ----------------------------------------------------------------

    train_docs = text_pipe("data/ag_news_csv/train.csv", 2, 0)
    test_docs = text_pipe("data/ag_news_csv/test.csv", 2, 0)

    dm = DataManager(train_docs, test_docs, config)
    dm.build_vocab(drop_less_than=config['data']['vocab_drop_threshold'])

    # optimizer, loss, metrics -------------------------------------------------
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
    )

    train_writer = tf.summary.create_file_writer(f'./logs/train/')
    val_writer  = tf.summary.create_file_writer(f'./logs/val/')

    nn = TextRNN(config, len(dm.vocab))
    nn.build_graph()
    train_loop(dm, model_id, 15, 0, False)
    print("saving model weights...")
    nn.save_weights(f"./checkpoints/{model_id}.h5")