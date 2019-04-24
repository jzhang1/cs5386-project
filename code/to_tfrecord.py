"""
Tokenize corpus into List[List[int]]
Convert this to examples (List[Int] of window_size, Int)
Store in TFRecord files of 100MB each
"""

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from corpus import Corpus
from create_tokenizer import load_tokenizer
import tensorflow as tf
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus_dir", dest="corpus_dir",
                        type=str, required=True)
    parser.add_argument("--tokenizer_file",
                        dest="tokenizer_file", type=str, required=True)
    parser.add_argument("--out_dir", dest="out_dir", type=str, required=True)
    parser.add_argument("--window_radius", dest="window_radius",
                        type=int, default=3, required=False)
    parser.add_argument("--file_limit", dest="file_limit",
                        type=int, default=1000000, required=False)
    parser.add_argument("--batch_size", dest="batch_size",
                        type=int, default=100000000, required=False)

    return parser.parse_args()


def rolling_window(a, window, step_size):
    shape = a.shape[:-1] + (a.shape[-1] - window + 2 - step_size, window)
    strides = a.strides + (a.strides[-1] * step_size,)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=False)


def sequence_to_cbow(seq, window_radius):
    pad_seq = np.pad(seq, (window_radius, window_radius), 'constant')
    window = rolling_window(pad_seq, window_radius * 2 + 1, 1)

    context_indices = np.arange(2 * window_radius) + np.pad(
        np.ones(window_radius, dtype=np.int), (window_radius, 0), 'constant')
    context = window[:, context_indices]
    target = window[:, window_radius]
    return context, target


def sliding_window_count(seq_length, window_size):
    return seq_length - window_size


def generate_dataset(tokenizer, corpus, window_radius=3):
    for seq in tokenizer.texts_to_sequences_generator(corpus):
        yield sequence_to_cbow(seq, window_radius)


def collect_dataset(context_target_generator, batch_size=None):
    """
    context_target_generator: Iterator[Context, Target]
    Context: np.array2D[int]
    Target: np.array[int]

    batch_size = length of numpy array to cutoff at
    """
    total_size = 0
    context_list = []
    target_list = []
    for context, target in context_target_generator:
        context_list.append(context)
        target_list.append(target)

        context_shape = context.shape
        target_shape = target.shape
        current_size = np.product(context_shape) + np.product(target_shape)
        total_size += current_size
        if batch_size and total_size > batch_size:
            total_size = 0
            yield np.concatenate(context_list, axis=0), np.concatenate(target_list)
            context_list.clear()
            target_list.clear()
    yield np.concatenate(context_list, axis=0), np.concatenate(target_list)

if __name__ == "__main__":
    args = parse_args()

    tokenizer = load_tokenizer(args.tokenizer_file)
    corpus = Corpus(args.corpus_dir, args.file_limit)

    context_target_generator = generate_dataset(
        tokenizer, corpus, args.window_radius)
    for index, (X, y) in enumerate(collect_dataset(context_target_generator, args.batch_size)):
        X = X.astype(np.int64)
        y = y.astype(np.int64)
        example = tf.train.Example(features=tf.train.Features(feature={
            "X_shape": tf.train.Feature(int64_list=tf.train.Int64List(value=list(X.shape))),
            "X": tf.train.Feature(int64_list=tf.train.Int64List(value=X.flatten())),
            "y": tf.train.Feature(int64_list=tf.train.Int64List(value=y))
        }))

        outpath = os.path.join(args.out_dir, "{0}.tfrecord".format(index))
        with tf.python_io.TFRecordWriter(outpath) as tfwriter:
            tfwriter.write(example.SerializeToString())