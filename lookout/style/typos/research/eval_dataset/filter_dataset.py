import os

import pandas
import spacy

from sourced.ml.algorithms import TokenParser


def filter_equal_identifiers(dataset: str, filtered_dataset: str) -> None:
    data = pandas.read_csv(dataset, header=0, usecols=[0, 1], names=["wrong", "correct"], keep_default_na=False)

    # Filter examples with equal splits
    tp = TokenParser(min_split_length=1, stem_threshold=400, single_shot=True, max_token_length=400,
                     attach_upper=True)
    data["wrong_split"] = data["wrong"].apply(lambda x: " ".join(tp.split(x)))
    data["correct_split"] = data["correct"].apply(lambda x: " ".join(tp.split(x)))
    data = data[data["wrong_split"] != data["correct_split"]]

    os.system("python3 -m spacy download en")
    nlp = spacy.load('en', disable=['parser', 'ner'])

    # Filter examples with equal lemmas
    def _lemmatize(token):
        lemm = nlp(token)
        if len(lemm) > 1 or lemm[0].lemma_ == "-PRON-" or (token[-2:] == "ss" and lemm[0].lemma_ == token[:-1]):
            return token
        return lemm[0].lemma_
    data["wrong_lem"] = data["wrong_split"].apply(lambda x: " ".join(_lemmatize(token) for token in x.split()))
    data["correct_lem"] = data["correct_split"].apply(lambda x: " ".join(_lemmatize(token) for token in x.split()))
    data = data[(data["wrong_lem"] != data["correct_lem"]) & (data["wrong_lem"] != data["correct_split"]) & (
         data["correct_lem"] != data["wrong_split"])]

    # Save new dataset
    whole_data = pandas.read_csv(dataset, header=0, keep_default_na=False)
    whole_data = whole_data.loc[data.index]
    whole_data.to_csv(filtered_dataset, compression="xz", index=False)
