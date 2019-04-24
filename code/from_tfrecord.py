import tensorflow as tf
import os



def extract_dataset(data_record):
    features = {
        "X_shape": tf.FixedLenFeature([2], tf.int64),
        "X": tf.VarLenFeature(tf.int64),
        "y": tf.VarLenFeature(tf.int64)
    }
    sample = tf.parse_single_example(data_record, features)
    X_shape = sample["X_shape"]
    X = tf.sparse_tensor_to_dense(sample["X"])
    y = tf.sparse_tensor_to_dense(sample["y"])

    return tf.reshape(X, X_shape), y

if __name__ == "__main__":
    for tfrecord_file in os.listdir("tfrecord"):
        path = os.path.join("tfrecord", tfrecord_file)

        dataset = tf.data.TFRecordDataset([path]).map(extract_dataset)

        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:
            try:
                while True:
                    data_record = sess.run(next_element)
                    print(data_record)
            except:
                pass