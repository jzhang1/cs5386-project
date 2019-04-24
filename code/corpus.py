import os
import spacy

class Corpus:
    def __init__(self, dir_path):
        """
        Creates a corpus from all files in a directory
        """
        self.dir_path = os.path.abspath(dir_path)

    def __iter__(self):
        en = spacy.load('en')
        for file in os.listdir(self.dir_path):
            filepath = os.path.join(self.dir_path, file)
            with open(filepath, mode = 'r', encoding = 'utf-8') as f:
                for index, sentence in enumerate(en(f.read()).sents):
                    if index == 0:
                        continue
                    yield str(sentence)