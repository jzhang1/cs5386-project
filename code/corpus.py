import os
import spacy
import tqdm

class Corpus:
    def __init__(self, dir_path, show_progress = True):
        """
        Creates a corpus from all files in a directory
        """
        self.dir_path = os.path.abspath(dir_path)
        self.show_progress = show_progress

    def __iter__(self):
        en = spacy.load('en')
        dir_list = os.listdir(self.dir_path)
        if self.show_progress:
            dir_list = tqdm.tqdm(dir_list)
        for file in dir_list:
            filepath = os.path.join(self.dir_path, file)
            with open(filepath, mode = 'r', encoding = 'utf-8') as f:
                for index, sentence in enumerate(en(f.read()).sents):
                    if index == 0:
                        continue
                    yield str(sentence)