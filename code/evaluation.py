import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import os
import argparse
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir', type=str, required=True)
    parser.add_argument('--v2v_path',dest='v2v_path',type=str, default='./word2vec.h5', required=False)
    parser.add_argument('--labeled_path',dest='labeled_path',type=str, default = './',required=False)
    parser.add_argument('--input_length', dest='input_length',type=int, default=500, required=False)
    parser.add_argument('--embedding_dim', dest='embedding_dim',type=int, default=100, required=False)
    parser.add_argument('--batch_size', dest='batch_size',type=int, default=128, required=False)
    parser.add_argument('--epochs', dest='epochs',type=int, default=3, required=False)
    parser.add_argument('--plot_model', dest='plot',type=bool, default=False, required=False)
    return parser.parse_args()

def extract_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        dataset =  pickle.load(f)
    return dataset[0], dataset[1]

def get_model_weights(path, layer_num = None):
    word2vec = load_model(path)
    embedding_layer = word2vec.layers[layer_num]
    weights = embedding_layer.get_weights()
    return weights

def get_labeled_weights(dir_path):
    phase1_path = os.path.join(dir_path,'phase1.hdf5')
    phase2_path = os.path.join(dir_path,'phase2.hdf5')
    phase1_weights = get_model_weights(phase1_path, layer_num=1)
    phase2_weights = get_model_weights(phase2_path,layer_num=2)
    arr = np.concatenate((phase1_weights[0],phase2_weights[0]),axis=1)
    return [arr]

def get_embedding_layer(weights, embedding_dim, input_len, trainable=False):
    embedding = Embedding(input_dim=weights[0].shape[0], output_dim=embedding_dim, input_length=input_len)
    embedding.build(input_shape=(input_len,))
    embedding.set_weights(weights)
    embedding.trainable = trainable
    return embedding

def get_model(embedding):
    model=Sequential()
    model.add(embedding)
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    args  = parse_args()
    
    train = extract_pickle(os.path.join(args.data_dir, 'imdb_train.pkl'))
    test = extract_pickle(os.path.join(args.data_dir, 'imdb_test.pkl'))
    x_train = sequence.pad_sequences(train[0], maxlen=args.input_length)
    x_test = sequence.pad_sequences(test[0], maxlen=args.input_length)
    y_train = train[1]
    y_train = np.array(y_train,dtype=np.int8)
    y_test = test[1]
    y_test = np.array(y_test,dtype=np.int8)

    weights_v2v = get_model_weights(path = args.v2v_path,layer_num=1)
    weights_labeled = get_labeled_weights(args.labeled_path)
    embedding_v2v = get_embedding_layer(weights_v2v, embedding_dim=args.embedding_dim, input_len = args.input_length, trainable=False)
    embedding_labeled = get_embedding_layer(weights_labeled, embedding_dim=args.embedding_dim, input_len = args.input_length, trainable=False)
    model_v2v = get_model(embedding_v2v)
    model_labeled = get_model(embedding_labeled)

    if args.plot == True:
        plot_model(model_v2v,to_file='eval_model.png',show_shapes=True,show_layer_names=False)

    batch_size = args.batch_size
    num_epochs = args.epochs

    if not os.path.exists('training_history'):
        os.makedirs('training_history')

    for i in range(3):
        hist_labeled = model_labeled.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=num_epochs)
        hist_v2v = model_v2v.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=num_epochs)
        with open('./training_history/training_hist{}.pkl'.format(i),'wb') as f:
            pickle.dump(hist_v2v.history,f)
            pickle.dump(hist_labeled.history,f)