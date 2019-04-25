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
                        type=int, default=2500000, required=False)

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
    return window[:, np.append(context_indices, window_radius)]


def sliding_window_count(seq_length, window_size):
    return seq_length - window_size


def generate_dataset(tokenizer, corpus, window_radius=3):
    for seq in tokenizer.texts_to_sequences_generator(corpus):
        seq = np.array(seq, dtype=np.int)
        for example in sequence_to_cbow(seq, window_radius):
            yield example[:-1], example[-1]


if __name__ == "__main__":
    args = parse_args()

    tokenizer = load_tokenizer(args.tokenizer_file)
    corpus = Corpus(args.corpus_dir, args.file_limit)

    tfwriter = None
    index = 0
    counter = 0

    try:
        for x, y in generate_dataset(tokenizer, corpus, args.window_radius):
            example = tf.train.Example(features=tf.train.Features(feature={
                "x": tf.train.Feature(int64_list=tf.train.Int64List(value=x)),
                "y": tf.train.Feature(int64_list=tf.train.Int64List(value=[y]))
            }))

            if not tfwriter or counter > args.batch_size:
                if tfwriter:
                    tfwriter.close()
                counter = 0
                outpath = os.path.join(args.out_dir, "{0}.tfrecord".format(index))
                tfwriter = tf.python_io.TFRecordWriter(outpath)
                index += 1

            tfwriter.write(example.SerializeToString())
            counter += 1
    finally:
        if tfwriter:
            tfwriter.close()
