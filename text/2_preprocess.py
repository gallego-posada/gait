import glob
import pickle

import bs4
import os
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from stopwords import stopwords
wnl = WordNetLemmatizer()

if __name__ == '__main__':
    with open('data/words', 'rb') as f:
        words = set(pickle.load(f))

    local_words = set()
    try:
        os.mkdir("data/EnglishProcessed/")
    except Exception as e:
        print("Did not create directory: {}".format(e))
    for article_fname in sorted(glob.glob('data/English/*.txt')):
        with open(article_fname, 'r', encoding="latin1") as f:
            print(article_fname)
            text = bs4.BeautifulSoup(f.read(), 'html5lib').text

        raw_tokens = [wnl.lemmatize(w.lower()) for s in sent_tokenize(text) for w in word_tokenize(s)]
        tokens = [w for w in raw_tokens if w not in stopwords]
        # print([w for w in raw_tokens if w in stopwords])
        proc_tokens = []
        for token in tokens:
            if any(c.isalpha() for c in token):
                if token in words:
                    proc_tokens.append(token)
                else:
                    token = token.lower()
                    if token in words:
                        proc_tokens.append(token)

        local_words.update(proc_tokens)
        processed = ' '.join(proc_tokens)
        with open(article_fname.replace('English', 'EnglishProcessed'), 'w') as f:
            print(article_fname.replace('English', 'EnglishProcessed'))
            print(processed, file=f)

    with open('data/news_words', 'wb') as f:
        pickle.dump(list(local_words), f)
    print('Done')
