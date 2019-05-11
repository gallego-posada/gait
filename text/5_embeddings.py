import io
import pickle

import numpy as np


def load_vectors(fname, valid_words):
    indices = {w: i for (i, w) in enumerate(valid_words)}
    word_set = set(valid_words)
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    _, _ = map(int, fin.readline().split())
    data = [None for _ in range(len(valid_words))]
    for line in fin:
        tokens = line.rstrip().split(' ')
        token = tokens[0]
        if token not in word_set:
            continue
        data[indices[token]] = list(map(float, tokens[1:]))
    return np.array(data)


if __name__ == '__main__':
    with open('data/news_words', 'rb') as f:
        words = pickle.load(f)
    vectors = load_vectors('data/wiki-news-300d-1M-subword.vec', words)
    print('Vectors shape:', vectors.shape)
    with open('data/news_vectors', 'wb') as f:
        pickle.dump(vectors, f)
