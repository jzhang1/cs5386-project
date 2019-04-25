"""
Train hand labelled embedding using CBOW model
"""

from typings import List
from tensorflow.keras.preprocessing.text import Tokenizer
import argparse
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding, Lambda, Dense, Concatenate
from tensorflow.keras import backend as K
from from_tfrecord import load_dataset
from create_tokenizer import load_tokenizer

# python code\train_labelled_embedding.py --tokenizer_file "tokenizer.pickle" --dataset_dir "tfrecord" --output_file "word2vec.h5"
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer_file", dest="tokenizer_file", type=str, required=True)
    parser.add_argument("--dataset_dir", dest="dataset_dir", type=str, required=True)
    parser.add_argument("--num_parallel_calls", dest="num_parallel_calls", type=int, required=False)
    parser.add_argument("--output_file", dest="output_file", type=str, required=True)
    parser.add_argument("--train_batch_size", dest="train_batch_size", type=int, default=32, required=False)
    parser.add_argument("--train_steps_per_epoch", dest="train_steps_per_epoch", type=int, default=128, required=False)
    parser.add_argument("--train_epochs", dest="train_epochs", type=int, default=20, required=False)
    parser.add_argument("--embedding_size", dest="embedding_size", type=int, default=100, required=False)
    parser.add_argument("--window_size", dest="window_size", type=int, default=6, required=False)

    return parser.parse_args()

"""
First predict the target word with the hand labelled data
Labelled (15 dimensions) => Target (softmax)
Labelled => Target + Context Embedding => Target (softmax)
Labelled + Context Embedding => Our Embedding
"""

def labels_to_target_model(n_labelled_dimensions: int, layer_sizes: List[int], vocab_size: int):
    """
    Given the hand labelled dimensions of the target word, predict the target word
    If the target word was not in the labelled set, set all hand labelled dimensions to 0
    n_labelled_dimensions: the number of labelled dimensions
    """
    model = Sequential()
    model.add(Input(input_shape=(None, n_labelled_dimensions)))
    for layer_size in layer_sizes:
        model.add(Dense(layer_size, activation='relu', kernel_initializer='glorot_normal'))
    model.add(Dense(vocab_size, activation='softmax'))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    return model

def residual_word2vec(prior_model, embedding_size:int, window_size: int):
    """
    Given the prediction from labels_to_target_model,
    combine this prediction with averaged Context Embedding
    to predict the target
    prior_model: Keras model that output softmax prediction of target word
    """
    prior_output = prior_model.output
    vocab_size = prior_output.shape[-1].value

    # Freeze all weights of prior model
    prior_model.trainable = False
    prior_model.compile()

    prior_input = Input((vocab_size,))
    
    context_input = Input((window_size, vocab_size, ))
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=window_size)(context_input)
    averaged_context = Lambda(lambda x: K.mean(x, axis=1))(embedding)

    concat_layer = Concatenate()([prior_input, averaged_context])
    output = Dense(vocab_size, activation='softmax')(concat_layer)

    model = Model(inputs = [prior_model.input, context_input], output = output)

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    return model

def word2vec(vocab_size: int, embedding_size: int, window_size: int):
    model = Sequential()

    model.add(Embedding(input_dim = vocab_size, output_dim = embedding_size, input_length = window_size))
    model.add(Lambda(lambda x: K.mean(x, axis=1), output_shape = (embedding_size, )))
    model.add(Dense(vocab_size, activation="softmax"))

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    return model

if __name__ == "__main__":
    args = parse_args()

    tokenizer = load_tokenizer(args.tokenizer_file)
    
    vocab_size = len(tokenizer.word_index) + 1
    embedding_size = args.embedding_size
    window_size = args.window_size

    model = word2vec(vocab_size, embedding_size, window_size)
    dataset = load_dataset(args.dataset_dir)

    model.fit(dataset.repeat().batch(args.train_batch_size).make_one_shot_iterator(), steps_per_epoch = args.train_steps_per_epoch, epochs = args.train_epochs)
    model.save(args.output_file)