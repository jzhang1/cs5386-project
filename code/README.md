# Instructions
0. Download the corpus and put all the text files in a folder "corpus"
1. Create a tokenizer from the corpus and save it in a file
   1. python create_tokenizer.py --vocab_size 10000 --corpus_dir "corpus" --tokenizer_file "tokenizer.pickle" [--file_limit 1000000]
2. Generate the TFRecord dataset files
   1. python to_tfrecord.py --corpus_dir "corpus" --tokenizer_file "tokenizer.pickle" --out_dir "tfrecord" [--file_limit 1000000] [--window_radius 3] [--batch_size 2500000]
3. Train Word2Vec model
   1. python train_word2vec.py --tokenizer_file "tokenizer.pickle" --dataset_dir "tfrecord" --output_file "word2vec.h5" --checkpoint_dir "checkpoints" --train_epochs 100000
   2. Additional parameters
      1. num_parallel_calls: number of tfrecord files to read in parallel. Set this to the number of CPU cores available on the machine
      2. train_batch_size: default is 32, can be set higher, maybe 128 if there is enough GPU memory
      3. train_steps_per_epoch: default is 128, can be set higher for more training stability. Use multiples of the train_batch_size
      4. train_epochs: default is 20. Set higher number to train more
      5. embedding_size: size of the output embedding, default is 100
      6. window_size: default is 6, don't change this
4. Extract embedding weights from hand labelled data
   1. python extract_embedding_from_labels.py --tokenizer_file "tokenizer.pickle" --labels_file "final_labels.csv" --out_file "labels_embedding.npy"
5. Train Labelled Embedding model
   1. Phase 1 training
      1. python train_labelled_embedding_1.py --tokenizer_file "tokenizer.pickle" --labels_embedding "labels_embedding.npy" --dataset_dir "tfrecord" --output_file "labelled_embedding.hdf5" --checkpoint_dir "phase1_checkpoints"
   2. Phase 2 training
      1. python train_labelled_embedding_2.py --tokenizer_file "tokenizer.pickle" --phase1_model "phase1.hdf5" --dataset_dir "tfrecord" --output_file "phase2.hdf5" --checkpoint_dir "phase2_checkpoints" --residual_embedding_size 73
