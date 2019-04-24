import os
import spacy
import tqdm

class Corpus:
    def __init__(self, dir_path, limit = None, show_progress = True):
        """
        Creates a corpus from all files in a directory
        """
        self.dir_path = os.path.abspath(dir_path)
        self.show_progress = show_progress
        self.limit = limit

    def __iter__(self):
        en = spacy.load('en')
        dir_list = os.listdir(self.dir_path)
        if self.show_progress:
            dir_list = tqdm.tqdm(dir_list, total=self.limit) if self.limit else tqdm.tqdm(dir_list)
        for index, file in enumerate(dir_list):
            if self.limit and index > self.limit:
                break
            filepath = os.path.join(self.dir_path, file)
            with open(filepath, mode = 'r', encoding = 'utf-8') as f:
                for index, sentence in enumerate(en(f.read()).sents):
                    if index == 0:
                        continue
                    yield str(sentence)