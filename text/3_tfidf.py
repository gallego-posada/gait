import glob
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer


if __name__ == '__main__':
    with open('data/news_words', 'rb') as f:
        words = pickle.load(f)
    fnames = sorted(glob.glob('data/EnglishProcessed/*.txt'))
    vectorizer = TfidfVectorizer(input='filename',
                                 lowercase=False,
                                 vocabulary=words,
                                 sublinear_tf=True,
                                 norm='l1',
                                 encoding="latin1",
                                 )
    X = vectorizer.fit_transform(fnames)
    with open('data/tfidf', 'wb') as f:
        pickle.dump(X, f)
