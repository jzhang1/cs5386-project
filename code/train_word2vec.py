"""
Train Word2Vec using CBOW model
"""

from tensorflow.keras.preprocessing.text import Tokenizer
import argparse
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Lambda, Dense, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from from_tfrecord import load_dataset
from create_tokenizer import load_tokenizer

# python code\train_word2vec.py --tokenizer_file "tokenizer.pickle" --dataset_dir "tfrecord" --output_file "word2vec.hdf5" --checkpoint_dir "checkpoints"
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer_file", dest="tokenizer_file", type=str, required=True)
    parser.add_argument("--dataset_dir", dest="dataset_dir", type=str, required=True)
    parser.add_argument("--output_file", dest="output_file", type=str, required=True)
    parser.add_argument("--checkpoint_dir", dest="checkpoint_dir", type=str, required=True)
    parser.add_argument("--train_batch_size", dest="train_batch_size", type=int, default=32, required=False)
    parser.add_argument("--train_steps_per_epoch", dest="train_steps_per_epoch", type=int, default=128, required=False)
    parser.add_argument("--train_epochs", dest="train_epochs", type=int, default=20, required=False)
    parser.add_argument("--embedding_size", dest="embedding_size", type=int, default=100, required=False)
    parser.add_argument("--window_size", dest="window_size", type=int, default=6, required=False)

    return parser.parse_args()

def word2vec(vocab_size: int, embedding_size: int, window_size: int):
    input = Input(shape = (window_size, ))
    embedding = Embedding(input_dim = vocab_size, output_dim = embedding_size, input_length = window_size, embeddings_initializer = 'he_normal')(input)
    averaged_context = Lambda(lambda x: K.mean(x, axis = 1), output_shape = (embedding_size, ))(embedding)
    output = Dense(vocab_size, activation = 'softmax', kernel_initializer = 'he_normal')(averaged_context)

    model = Model(inputs = input, outputs = output)

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

    return model

if __name__ == "__main__":
    args = parse_args()

    tokenizer = load_tokenizer(args.tokenizer_file)
    
    vocab_size = min(tokenizer.num_words, len(tokenizer.word_index)) + 1
    embedding_size = args.embedding_size
    window_size = args.window_size

    model = word2vec(vocab_size, embedding_size, window_size)

    dataset = load_dataset(args.dataset_dir)

    checkpoint_path = os.path.join(args.checkpoint_dir, "word2vec.{epoch:02d}-{loss:.2f}.hdf5")
    callbacks_list = [
        EarlyStopping(monitor='loss', min_delta=0.0001, patience=100, mode='min', verbose=1),
        ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    ]

    model.fit(dataset.repeat().batch(args.train_batch_size).make_one_shot_iterator(), steps_per_epoch = args.train_steps_per_epoch, epochs = args.train_epochs, callbacks = callbacks_list)
    model.save(args.output_file)