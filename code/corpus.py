import os
import spacy
import tqdm

class Corpus:
    def __init__(self, dir_path, corpus_list = None, limit = None, show_progress = True):
        """
        Creates a corpus from all files in a directory
        corpus_list: file path to file listing all files in corpus in dir_path
        """
        self.dir_path = os.path.abspath(dir_path)
        self.corpus_list = corpus_list
        self.show_progress = show_progress
        self.limit = limit

    def __iter__(self):
        en = spacy.load('en')
        if self.corpus_list:
            with open(self.corpus_list, mode = 'r', encoding = 'utf-8') as corpus_list_file:
                dir_list = [filename.strip() for filename in corpus_list_file]
        else:
            dir_list = os.listdir(self.dir_path)\
                
        if self.limit:
            if self.limit < len(dir_list):
                limit = self.limit
                dir_list = dir_list[:limit]
            else:
                limit = None
        if self.show_progress:
            dir_list = tqdm.tqdm(dir_list)
        for index, file in enumerate(dir_list):
            filepath = os.path.join(self.dir_path, file)
            with open(filepath, mode = 'r', encoding = 'utf-8') as f:
                for index, sentence in enumerate(en(f.read()).sents):
                    if index == 0:
                        continue
                    yield str(sentence)