"""
App to help speed up labelling of items
"""

import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", dest="input",
                        type=str, default="Top5000.csv")
    parser.add_argument("--range", dest="range", type=int,
                        nargs=2, default=[101, 550])
    parser.add_argument("--output", dest="output",
                        type=str, default="output.csv")

    return parser.parse_args()


def parse_binary(value):
    if value == None:
        return 0
    return -1 if value <= 0 else 1


def parse_category(categories, default_value):
    def parse(value):
        if value == None:
            return default_value
        return categories[value]
    return parse


parse_trinary = parse_category([-1, 0, 1], 0)

fields = {
    "Person?": parse_binary,
    "Object?": parse_binary,
    "Plants?": parse_binary,
    "Animals?": parse_binary,
    "Location?": parse_binary,
    "Time?": parse_binary,
    "Process?": parse_binary,
    "Action?": parse_binary,
    "Quality?": parse_binary,
    "Superlative[0,0.3,0.6,1]": parse_category([0.0, 0.3, 0.6, 1.0], 0.0),
    "Negative/Neutral/Positive[-1,0,1]": parse_trinary,
    "Male/Female": parse_binary,
    "Concrete/Abstract": parse_binary
}

input_map = {
    "a": None,
    "z": 0,
    "x": 1,
    "c": 2,
    "v": 3
}

pos_map = {
    'a': "article",
    'v': "verb",
    'c': "conjunction",
    'i': "preposition",
    't': "to",
    'p': "possessive",
    'd': "determiner",
    'x': "negative",
    'r': "adverb",
    'm': "numeric",
    'n': "noun",
    'e': "there",
    'j': "adjective",
    'u': "exclaimation"
}


def label_word(word: str, pos: str):
    word_labels = {}
    for field in fields:
        parser = fields[field]

        input_valid = False
        while not input_valid:
            value = input("{0} ({1}) is {2} ".format(
                word, pos_map[pos], field))
            if value in input_map:
                input_valid = True
                mapped_value = input_map[value]
                parsed_value = parser(mapped_value)
                word_labels[field] = parsed_value
    return word_labels


def manually_label():
    args = parse_args()

    table = pd.read_csv(args.input)
    table = table[["Rank", "Word", "POS"]]

    [range_min, range_max] = args.range
    table = table[(table["Rank"] >= range_min) & (table["Rank"] <= range_max)]

    # add new columns from fields
    for field in fields:
        # initialize column to default value 0
        table[field] = 0

    count = 0
    for row in table.iterrows():
        index, values = row
        word = values["Word"]
        pos = values["POS"]

        print("==={0}. {1}===".format(count, word))
        count += 1

        word_labels = label_word(word, pos)
        for field in word_labels:
            table.loc[index, field] = word_labels[field]

    table.to_csv(args.output)

def extract_450():
    args = parse_args()

    table = pd.read_csv(args.input)
    table = table[["Rank", "Word", "POS"]]

    [range_min, range_max] = args.range
    table = table[(table["Rank"] >= range_min) & (table["Rank"] <= range_max)]

    # add new columns from fields
    for field in fields:
        # initialize column to default value 0
        table[field] = 0

    table = table.sort_values("POS")

    table.to_csv(args.output)

def postprocess_table():
    args = parse_args()

    table = pd.read_csv(args.input)

    for field in fields:
        field_parser = fields[field]
        for index, value in enumerate(table[field]):
            try:
                float(value)
            except:
                value = input_map[value]
                table.loc[index, field] = field_parser(value)
    
    table = table.sort_values("Rank")
    table.to_csv(args.output)

if __name__ == "__main__":
    postprocess_table()
    

# our categories cannot capture a lot of meaning
# things like adjectives and adverbs or articles are not well described

# Quantity (one, two, three, most, least, many, some, every)
# Sequence (first, last, 3rd)
# Is Event?
# Age
# Distance

# Express word relations
# opposite meaning
# relative degree