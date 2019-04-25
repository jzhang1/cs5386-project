import tensorflow as tf
import os

def extract_dataset(data_record, window_size = 6):
    features = {
        "x": tf.FixedLenFeature([window_size], tf.int64),
        "y": tf.FixedLenFeature([1], tf.int64)
    }
    sample = tf.parse_single_example(data_record, features)
    return sample["x"], sample["y"][0]

def load_dataset(dataset_dir, num_parallel_calls = None):
    dataset_files = [os.path.join(dataset_dir, filename) for filename in os.listdir(dataset_dir)]
    return tf.data.TFRecordDataset(dataset_files).map(extract_dataset, num_parallel_calls = num_parallel_calls)

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