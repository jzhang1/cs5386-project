from train_labelled_embedding_1 import residual_word2vec
from from_tfrecord import load_dataset
from create_tokenizer import load_tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import argparse
import os

# labelled dimensions = 27, to keep total embedding size = 100, the residual dimensions are 73
# python code\train_labelled_embedding_2.py --tokenizer_file "tokenizer.pickle" --phase1_model "phase1.hdf5" --dataset_dir "tfrecord" --output_file "phase2.hdf5" --checkpoint_dir "phase2_checkpoints" --residual_embedding_size 73

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer_file",
                        dest="tokenizer_file", type=str, required=True)
    parser.add_argument("--dataset_dir", dest="dataset_dir",
                        type=str, required=True)
    parser.add_argument("--phase1_model", dest="phase1_model",
                        type=str, required=True)
    parser.add_argument("--output_file", dest="output_file",
                        type=str, required=True)
    parser.add_argument("--checkpoint_dir",
                        dest="checkpoint_dir", type=str, required=True)
    parser.add_argument("--train_batch_size", dest="train_batch_size",
                        type=int, default=32, required=False)
    parser.add_argument("--train_steps_per_epoch",
                        dest="train_steps_per_epoch", type=int, default=128, required=False)
    parser.add_argument("--train_epochs", dest="train_epochs",
                        type=int, default=20, required=False)
    parser.add_argument("--residual_embedding_size",
                        dest="residual_embedding_size", type=int, default=100, required=False)
    parser.add_argument("--window_size", dest="window_size",
                        type=int, default=6, required=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tokenizer = load_tokenizer(args.tokenizer_file)
    dataset = load_dataset(args.dataset_dir)

    # Phase 2
    phase1_model = load_model(args.phase1_model)
    phase2_model = residual_word2vec(
        phase1_model, args.residual_embedding_size, args.window_size)

    checkpoint_path = os.path.join(
        args.checkpoint_dir, "labelled_embedding_2.{epoch:02d}-{loss:.2f}.hdf5")
    callbacks_list = [
        EarlyStopping(monitor='loss', min_delta=0.0001,
                      patience=100, mode='min', verbose=1),
        ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1,
                        save_best_only=True, mode='min', save_weights_only=True)
    ]

    phase2_model.fit(dataset.repeat().batch(args.train_batch_size).make_one_shot_iterator(),
                     steps_per_epoch=args.train_steps_per_epoch,
                     epochs=args.train_epochs,
                     callbacks=callbacks_list)
    phase2_model.save(args.output_file)
