import io
import pickle


def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    # data = {}
    data = set()
    for line in fin:
        # tokens = line.rstrip().split(' ')
        # data[tokens[0]] = map(float, tokens[1:])
        word = line.rstrip().split(' ', 1)[0]
        data.add(word)
    return data


if __name__ == '__main__':
    vectors = load_vectors('data/wiki-news-300d-1M-subword.vec')
    with open('data/words', 'wb') as f:
        pickle.dump(list(vectors), f)
