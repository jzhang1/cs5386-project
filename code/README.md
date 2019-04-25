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