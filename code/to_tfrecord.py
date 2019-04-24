"""
Tokenize corpus into List[List[int]]
Convert this to examples (List[Int] of window_size, Int)
Store in TFRecord files of 100MB each
"""

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from corpus import Corpus
from create_tokenizer import load_tokenizer
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus_dir", dest="corpus_dir", type=str, required=True)
    parser.add_argument("--tokenizer_file", dest="tokenizer_file", type=str, required=True)
    parser.add_argument("--out_dir", dest="out_dir", type=str, required=True)

    return parser.parse_args()

def rolling_window(a, window, step_size):
    shape = a.shape[:-1] + (a.shape[-1] - window + 2 - step_size, window)
    strides = a.strides + (a.strides[-1] * step_size,)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides, writeable=False)

def sequence_to_cbow(seq, window_radius):
    pad_seq = np.pad(seq, (window_radius, window_radius), 'constant')
    window = rolling_window(pad_seq, window_radius * 2 + 1, 1)
    
    context_indices = np.arange(2 * window_radius) + np.pad(np.ones(window_radius, dtype=np.int), (window_radius, 0), 'constant')
    context = window[:, context_indices]
    target = window[:, window_radius]
    return context, target

def sliding_window_count(seq_length, window_size):
    return seq_length - window_size

def generate_dataset(tokenizer, corpus, window_radius = 3):
    for seq in tokenizer.texts_to_sequences_generator(corpus):
        yield sequence_to_cbow(seq, window_radius)

def collect_dataset(context_target_generator, batch_size = 4000000):
    """
    context_target_generator: Iterator[Context, Target]
    Context: np.array2D[int]
    Target: np.array[int]

    batch_size = length of numpy array to cutoff at
    """
    for context, target in context_target_generator:
        context_shape = context.shape
        target_shape = target.

if __name__ == "__main__":
    args = parse_args()

    tokenizer = load_tokenizer(args.tokenizer_file)
    corpus = Corpus(args.corpus_dir)


