import pandas as pd
import numpy as np
import pickle
from create_tokenizer import load_tokenizer
import argparse
import os

# python extract_embedding_from_labels.py --tokenizer_file "../tokenizer.pickle" --labels_file "final_labels.csv" --out_file "../labels_embedding.npy"
def table_columns(table):
    return [column for column in table.columns.values if column not in {"Rank", "Word"}]

def extract_table_values(table, columns):
    return table[columns].values

def generate_embedding_for_tokenizer(tokenizer, labels):
    columns = table_columns(labels)
    embedding = np.zeros((tokenizer.num_words, len(columns)))
    label_indices = np.array([tokenizer.word_index[word] for word in labels["Word"]])
    embedding[label_indices,:] = extract_table_values(labels, columns)
    return embedding

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer_file",
                        dest="tokenizer_file", type=str, required=True)
    parser.add_argument("--labels_file", dest="labels_file", type=str, required=True)
    parser.add_argument("--out_file", dest="out_file", type=str, required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    tokenizer = load_tokenizer(args.tokenizer_file)
    labels = pd.read_csv(args.labels_file)
    embedding = generate_embedding_for_tokenizer(tokenizer, labels)
    np.save(args.out_file, embedding, allow_pickle = False)