import glob
import pickle

import numpy as np
import torch
from torch import nn, optim


def print_summary(words, probs, embs):
    print('\nSummary: [to do]')


if __name__ == '__main__':
    with open('data/news_words', 'rb') as f:
        all_words = pickle.load(f)
    with open('data/tfidf', 'rb') as f:
        tdidf_articles = pickle.load(f)
    with open('data/news_vectors', 'rb') as f:
        word_embs = pickle.load(f)

    fnames = sorted(glob.glob('data/EnglishProcessed/*.txt'))
    for i, article_fname in enumerate(fnames[:10]):
        feature_index = tdidf_articles[i, :].nonzero()[1]
        if not feature_index.size:
            continue

        with open(article_fname, 'r') as f:
            article = f.read().strip()
        print(article)
        print('-----')
        words = [all_words[x] for x in feature_index]
        probs = [tdidf_articles[i, x] for x in feature_index]
        assert np.isclose(np.sum(probs), 1.0)
        embs = [word_embs[x] for x in feature_index]
        inp = sorted(list(zip(probs, words)), reverse=True)
        print('\nInput:', ['%s %0.2f' % (w, p * 100) for (p, w) in inp])
        print_summary(words, probs, embs)
        print("===============")
        print('\n')
