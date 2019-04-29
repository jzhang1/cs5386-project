"""
Train hand labelled embedding using CBOW model

First predict the target word with the hand labelled data
Labelled (dimensions) => Target (softmax)
Labelled => Target + Context Embedding => Target (softmax)
Labelled + Context Embedding => Our Embedding
"""

from typing import List
from tensorflow.keras.preprocessing.text import Tokenizer
import argparse
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding, Lambda, Dense, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from from_tfrecord import load_dataset
from create_tokenizer import load_tokenizer

# python code\train_labelled_embedding_1.py --tokenizer_file "tokenizer.pickle" --labels_embedding "labels_embedding.npy" --dataset_dir "tfrecord" --output_file "phase1.hdf5" --checkpoint_dir "phase1_checkpoints"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer_file",
                        dest="tokenizer_file", type=str, required=True)
    parser.add_argument("--labels_embedding",
                        dest="labels_embedding", type=str, required=True)
    parser.add_argument("--dataset_dir", dest="dataset_dir",
                        type=str, required=True)
    parser.add_argument("--output_file", dest="output_file",
                        type=str, required=True)
    parser.add_argument("--checkpoint_dir", dest="checkpoint_dir", type=str, required=True)
    parser.add_argument("--train_batch_size", dest="train_batch_size",
                        type=int, default=32, required=False)
    parser.add_argument("--train_steps_per_epoch",
                        dest="train_steps_per_epoch", type=int, default=128, required=False)
    parser.add_argument("--train_epochs", dest="train_epochs",
                        type=int, default=20, required=False)
    parser.add_argument("--window_size", dest="window_size",
                        type=int, default=6, required=False)

    return parser.parse_args()


def labels_to_target_model(embedding_weights, layer_sizes: List[int], window_size: int):
    """
    embedding_weights: numpy array of (vocab_size, labelled_dimensions)
    """
    vocab_size, n_labelled_dimensions = embedding_weights.shape
    weights = np.zeros((vocab_size + 1, n_labelled_dimensions))
    weights[1:, :] = embedding_weights

    input = Input(shape=(window_size, ))
    embedding = Embedding(vocab_size + 1, n_labelled_dimensions,
                          weights=[weights], input_length=window_size, trainable=False)(input)
    averaged_context = Lambda(lambda x: K.mean(x, axis = 1), output_shape = (n_labelled_dimensions, ))(embedding)
    x = averaged_context
    for layer_size in layer_sizes:
        x = Dense(layer_size, activation='relu',
                  kernel_initializer='he_normal')(x)
    output = Dense(vocab_size, activation='softmax',
                   kernel_initializer='he_normal')(x)

    model = Model(inputs=input, outputs=output)

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    return model


def residual_word2vec(prior_model, embedding_size: int, window_size: int):
    """
    Given the prediction from labels_to_target_model,
    combine this prediction with averaged Context Embedding to predict the target
    prior_model: Keras model that output softmax prediction of target word
    """
    input = Input(shape=(window_size, ))
    prior_output = prior_model(input)
    vocab_size = prior_output.shape[-1].value

    # Freeze all weights of prior model
    for layer in prior_model.layers:
        layer.trainable = False
    prior_model.trainable = False
    prior_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    embedding = Embedding(input_dim=vocab_size + 1, output_dim=embedding_size,
                          input_length=window_size, embeddings_initializer='he_normal')(input)
    averaged_context = Lambda(lambda x: K.mean(
        x, axis=1), output_shape=(embedding_size, ))(embedding)

    # Ideally we can use prior_output directly in Concatenate. Unfortunately having a
    # vocabulary size of 10,000 and an output of 10,000 will lead to 100,000,000 parameters
    # which will bog down GPU RAM during training
    # We summarize the output of Phase 1 model into 100 dimensions here to reduce total weight count
    # to around 1 million
    prior_summary = Dense(100, activation='relu', kernel_initializer='he_normal')(prior_output)

    joined_input = Concatenate()([prior_summary, averaged_context])
    output = Dense(vocab_size, activation='softmax',
                   kernel_initializer='he_normal')(joined_input)

    model = Model(inputs=input, outputs=output)

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    # print(model.summary())

    return model

"""
Training Phase 1
Training Phase 2
Extract embedding weights from Embedding Layer = ResidualEmbedding
Our Embedding = Concat(Labels, ResidualEmbedding)
"""
if __name__ == "__main__":
    args = parse_args()

    tokenizer = load_tokenizer(args.tokenizer_file)

    vocab_size = tokenizer.num_words
    window_size = args.window_size
    labels_embedding = args.labels_embedding

    embedding_weights = np.load(labels_embedding)

    dataset = load_dataset(args.dataset_dir)

    # Phase 1
    phase1_layers = []
    phase1_model = labels_to_target_model(
        embedding_weights, phase1_layers, window_size)

    checkpoint_path = os.path.join(args.checkpoint_dir, "labelled_embedding_1.{epoch:02d}-{loss:.2f}.hdf5")
    callbacks_list = [
        EarlyStopping(monitor='loss', min_delta=0.0001, patience=100, mode='min', verbose=1),
        ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    ]

    phase1_model.fit(dataset.repeat().batch(args.train_batch_size).make_one_shot_iterator(), 
        steps_per_epoch = args.train_steps_per_epoch,
        epochs = args.train_epochs,
        callbacks = callbacks_list)
    phase1_model.save(args.output_file)