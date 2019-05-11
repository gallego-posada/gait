import glob
import pickle

import numpy as np


if __name__ == '__main__':
    with open('data/news_words', 'rb') as f:
        words = pickle.load(f)
    np_words = np.array(words)
    with open('data/tfidf', 'rb') as f:
        X = pickle.load(f)

    fnames = sorted(glob.glob('data/EnglishProcessed/*.txt'))
    for i, article_fname in enumerate(fnames[:10]):
        with open(article_fname, 'r') as f:
            article = f.read()
        print(article)
        print('---')

        feature_index = X[i, :].nonzero()[1]
        scores = [X[i, x] for x in feature_index]
        print('Sum of scores:', np.sum(scores))
        print('---')
        doc = []
        for idx, score in zip(feature_index, scores):
            doc.append((score, words[idx]))
        doc.sort(reverse=True)
        print([(w, int(s * 10000)) for (s, w) in doc])
        print()
        print("===============")
        print()
