import io
import pickle

import numpy as np
import umap


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


def vectors_to_tsne(vectors):
    reducer = umap.UMAP(n_components=10)
    trans = reducer.fit_transform(vectors)

    return trans


if __name__ == '__main__':
    with open('data/news_words', 'rb') as f:
        words = pickle.load(f)
    vectors = load_vectors('data/wiki-news-300d-1M-subword.vec', words)
    transformed = vectors_to_tsne(vectors)
    print('Vectors shape:', vectors.shape)
    print('Transformed shape:', transformed.shape)
    with open('data/news_vectors', 'wb') as f:
        pickle.dump(vectors, f)

    with open('data/transformed_vectors', 'wb') as f:
        pickle.dump(transformed, f)
