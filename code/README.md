# Instructions
0. Download the corpus and put all the text files in a folder "corpus"
1. Create a tokenizer from the corpus and save it in a file
   1. python create_tokenizer.py --vocab_size 10000 --corpus_dir "corpus" --tokenizer_file "tokenizer.pickle" [--file_limit 1000000]
2. Generate the TFRecord dataset files
   1. python to_tfrecord.py --corpus_dir "corpus" --tokenizer_file "tokenizer.pickle" --out_dir "tfrecord" [--file_limit 1000000] [--window_radius 3] [--batch_size 1000000]