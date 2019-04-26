import argparse
from create_tokenizer import load_tokenizer
import numpy as np
import os
import sklearn
from sklearn.datasets import load_files
import tensorflow as tf
from tqdm import tqdm
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imdb_dir", dest="imdb_dir", type=str, required=True)
    parser.add_argument("--tokenizer_file", dest="tokenizer_file", type=str, required=True)
    parser.add_argument("--out_dir", dest="out_dir", type=str, required=True)
    return parser.parse_args()

args  = parse_args()

train = os.path.join(args.imdb_dir, 'train')
test = os.path.join(args.imdb_dir, 'test')

def load_data():
    movie_train = load_files(train, shuffle=True)
    movie_test = load_files(test, shuffle=True)
    return movie_train, movie_test

def generate_dataset(tokenizer, dataset):
    docs = dataset.data
    lables = dataset.target
    x_y = zip(docs,lables)
    for data in x_y:
        text = str(data[0])
        doc = tokenizer.texts_to_sequences([text])[0]
        doc = np.array(doc, dtype=np.int)  
        yield doc, data[1]

def to_tfrecord(tokenizer, dataset, output_name):
    outpath = os.path.join(args.out_dir,'{}.tfrecord'.format(output_name))
    tfwriter = tf.python_io.TFRecordWriter(outpath)
    try:
        for x, y in tqdm(generate_dataset(tokenizer, dataset)):
            example = tf.train.Example(features=tf.train.Features(feature={
                "x": tf.train.Feature(int64_list=tf.train.Int64List(value=x)),
                "y": tf.train.Feature(int64_list=tf.train.Int64List(value=[y]))
            }))
            tfwriter.write(example.SerializeToString())
    finally:
        if tfwriter:
            tfwriter.close()

def to_pickle(tokenizer, dataset, output_name):
    l_x = []
    l_y = []
    for x, y in tqdm(generate_dataset(tokenizer, dataset)):
        l_x.append(x)
        l_y.append(y)
    outpath = os.path.join(args.out_dir,'{}.pkl'.format(output_name))
    data = (l_x,l_y)
    with open (outpath,'wb') as f:
        pickle.dump(data,f)

if __name__ == "__main__":
    train_set, test_set = load_data()
    print('Loading dataset done.')
    tokenizer = load_tokenizer(args.tokenizer_file)
    to_pickle(tokenizer,train_set,'imdb_train')
    to_pickle(tokenizer,test_set,'imdb_test')
