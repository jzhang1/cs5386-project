import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import os
import argparse
import pickle
import numpy as np
from tensorflow.keras.models import load_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir', type=str, required=True)
    parser.add_argument('--model_path',dest='model_path',type=str, required=True)
    parser.add_argument('--input_length', dest='input_length',type=int, default=500, required=False)
    parser.add_argument('--embedding_dim', dest='embedding_dim',type=int, default=100, required=False)
    parser.add_argument('--batch_size', dest='batch_size',type=int, default=128, required=False)
    parser.add_argument('--epochs', dest='epochs',type=int, default=3, required=False)
    return parser.parse_args()

args  = parse_args()

def extract_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        dataset =  pickle.load(f)
    return dataset[0], dataset[1]

def get_embedding_layer(vocab_size=10001, embedding_dim=args.embedding_dim, input_len = args.input_length, trainable=False):
    word2vec = load_model(args.model_path)
    embedding_layer = word2vec.layers[1]
    weights = embedding_layer.get_weights()
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_len)
    embedding.build(input_shape=(input_len,))
    embedding.set_weights(weights)
    embedding.trainable = trainable
    return embedding

def get_model():
    embedding = get_embedding_layer()
    model=Sequential()
    model.add(embedding)
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    train = extract_pickle(os.path.join(args.data_dir, 'imdb_train.pkl'))
    test = extract_pickle(os.path.join(args.data_dir, 'imdb_test.pkl'))
    x_train = sequence.pad_sequences(train[0], maxlen=args.input_length)
    x_test = sequence.pad_sequences(test[0], maxlen=args.input_length)
    y_train = train[1]
    y_train = np.array(y_train,dtype=np.int8)
    y_test = test[1]
    y_test = np.array(y_test,dtype=np.int8)

    model = get_model()
    batch_size = args.batch_size
    num_epochs = args.epochs
    x_valid, y_valid = x_train[:batch_size], y_train[:batch_size]
    x_train2, y_train2 = x_train[batch_size:], y_train[batch_size:]
    model.fit(x_train2, y_train2, validation_data=(x_valid, y_valid), batch_size=batch_size, epochs=num_epochs)

    scores = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', scores[1])