#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import csv
from math import ceil
import multiprocessing as mp
from typing import List, Tuple
import spacy

CPUS = mp.cpu_count() - 1
CONTRACTION_DICT = {
    "ain't" : "is not", 
    "aren't" : "are not",
    "can't" : "cannot", 
    "could've" : "could have", 
    "couldn't" : "could not", 
    "didn't" : "did not",  
    "doesn't" : "does not", 
    "don't" : "do not", 
    "hadn't" : "had not", 
    "hasn't" : "has not", 
    "haven't" : "have not", 
    "he'd" : "he would",
    "he'll" : "he will", 
    "he's" : "he is", 
    "she'd" : "she would", 
    "she'll" : "she will", 
    "she's" : "she is", 
    "how'd" : "how did", 
    "how'll" : "how will", 
    "how's" : "how is",  
    "i'd" : "i would", 
    "i'll" : "i will",  
    "i'm" : "i am", 
    "i've" : "i have", 
    "isn't" : "is not", 
    "it'd" : "it would", 
    "it'll" : "it will", 
    "it's" : "it is", 
    "let's" : "let us", 
    "mayn't" : "may not", 
    "might've" : "might have",  
    "mustn't" : "must not", 
    "needn't" : "need not", 
    "shan't" : "shall not", 
    "should've" : "should have", 
    "shouldn't" : "should not", 
    "that'd" : "that would", 
    "that's" : "that is", 
    "here's" : "here is",
    "there's" : "there is",
    "they'd" : "they would", 
    "they'll" : "they will", 
    "they're" : "they are", 
    "they've" : "they have", 
    "wasn't" : "was not", 
    "we'd" : "we would", 
    "we'll" : "we will", 
    "we're" : "we are", 
    "we've" : "we have", 
    "weren't" : "were not", 
    "what'll" : "what will", 
    "what're" : "what are",  
    "what's" : "what is", 
    "when's" : "when is", 
    "where'd" : "where did", 
    "where's" : "where is", 
    "who'll" : "who will", 
    "who's" : "who is", 
    "why've" : "why have", 
    "won't" : "will not", 
    "would've" : "would have", 
    "wouldn't" : "would not", 
    "y'all" : "you all", 
    "you'd" : "you would", 
    "you'll" : "you will", 
    "you're" : "you are", 
    "you've" : "you have",
}

def text_pipe(file_path: str, text_col: int, label_col: int) -> List[str]:

    """
    Text processing pipeline. Distributes processing among available CPUs  
    """

    assert isinstance(file_path, str), "'file_path' needs to be a str type"
    assert file_path.split(".")[-1] == "csv", "'file_path' should be a csv file"
    print("1) Loading CSV file.", end=" ")

    raw = []
    sample_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            label = row[label_col]
            text = str(row[text_col])
            raw.append(
                (' '.join(text.split()), label)
            )
            sample_count += 1

    print("2) Processing text.", end=" ")
    batch_size = ceil(sample_count / CPUS)
    mp_args = [(raw[i * batch_size : (i + 1) * batch_size], 
        spacy.load("en_core_web_sm")) for i in range(CPUS)
    ]
    pool = mp.Pool(processes=CPUS)
    res = pool.starmap(process_batch, mp_args)
    pool.close()
    pool.join()

    print("3) Text processing complete.")
    return [sample for group in res for sample in group]

def process_batch(samples: List[Tuple[str, int]], nlp) -> List[Tuple[str, int]]:
    
    """
    Processing func to be distributed
    """

    processed = []
    for sample in samples:
        text = sample[0].strip().lower()
        text = re.sub(r"’|`", "'", text)
        text = re.sub(r"( #39;)|( #145;)|( #146;)", "'", text)
        text = re.sub(r"( #147;)|( #148;)", '"', text)  

        for k, v in CONTRACTION_DICT.items():
            text = re.sub(k, v, text)

        text = re.sub(r"(&lt;).*?(&gt;)", " ", text)
        text = re.sub("[^a-z\d]", " ", text)
        text = re.sub(r"[0-9]+", "#", text)

        tokens = nlp(' '.join(text.split()))
        lemmas = [tok.lemma_ for tok in tokens if not tok.is_stop]
        
        processed.append(
            (' '.join(lemmas), sample[1])
        )

    return processed

if __name__ == "__main__":

    cleaned = text_pipe("../data/ag_news_csv/test.csv", 2, 0)
    print(cleaned[1])
    f = open("../data/tmp.txt", 'w')
    data = ''
    for row in cleaned:
        data += row[1] + ' ' + row[0] + '\n'
        
    f.write(data)
    f.close()
