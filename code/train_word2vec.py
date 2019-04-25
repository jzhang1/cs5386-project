"""
Load dataset from TFRecord and train embedding
"""

from tensorflow.keras.preprocessing.text import Tokenizer
import argparse
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Lambda, Dense
from tensorflow.keras import backend as K
from from_tfrecord import extract_dataset
from create_tokenizer import load_tokenizer

# python code\train_word2vec.py --tokenizer_file "tokenizer.pickle" --dataset_dir "tfrecord" --output_file "word2vec.h5"
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

def load_dataset(dataset_dir, num_parallel_calls = None):
    dataset_files = [os.path.join(dataset_dir, filename) for filename in os.listdir(dataset_dir)]
    return tf.data.TFRecordDataset(dataset_files).map(extract_dataset, num_parallel_calls = num_parallel_calls)


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