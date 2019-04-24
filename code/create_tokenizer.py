from tensorflow.keras.preprocessing.text import Tokenizer
from corpus import Corpus
import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab_size", dest="vocab_size", type=int, required=True)
    parser.add_argument("--corpus_dir", dest="corpus_dir", type=str, required=True)
    parser.add_argument("--tokenizer_file", dest="tokenizer_file", type=str, required=True)

    return parser.parse_args()

def load_tokenizer(filepath):
    with open(filepath, mode = 'rb') as file:
        return pickle.load(file)

if __name__ == "__main__":
    args = parse_args()

    corpus = Corpus(args.corpus_dir)

    tokenizer = Tokenizer(num_words = args.vocab_size)
    tokenizer.fit_on_texts(corpus)

    with open(args.tokenizer_file, mode = 'wb') as outfile:
        pickle.dump(tokenizer, outfile)
