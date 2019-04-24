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

def parse_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--vocab_size", dest="vocab_size", type=int, required=True)
    # parser.add_argument("--corpus_dir", dest="corpus_dir", type=str, required=True)
    # parser.add_argument("--tokenizer_file", dest="tokenizer_file", type=str, required=True)

    parser.add_argument("--tokenizer_file", dest="tokenizer_file", type=str, required=True)
    parser.add_argument("--dataset_dir", dest="dataset_dir", type=str, required=True)
    parser.add_argument("--num_parallel_calls", dest="num_parallel_calls", type=int, required=False)

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

def train(model, dataset, batch_size = 32):
    estimator = tf.keras.estimator.model_to_estimator(keras_model=model)
    estimator.train(input_fn = lambda: dataset.repeat().batch(batch_size), max_steps = 19)
    return estimator

if __name__ == "__main__":
    args = parse_args()

    tokenizer = load_tokenizer(args.tokenizer_file)
    
    vocab_size = len(tokenizer.word_index) + 1
    embedding_size = 100
    window_size = 6
    model = word2vec(vocab_size, embedding_size, window_size)
    dataset = load_dataset(args.dataset_dir)
    estimator = train(model, dataset)

    # estimator.export_saved_model
