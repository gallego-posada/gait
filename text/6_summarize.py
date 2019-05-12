from collections import deque
import glob
import pickle
import sys
sys.path.append('..')

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from renyi import mink_sim_divergence
import utils


def print_summary(words, probs, embs):
    p = torch.tensor(np.array(probs, dtype=np.float32)[None, ...])
    # norm = torch.norm(embs, p=2, dim=1, keepdim=True)
    # cosine = (embs @ embs.t()) / norm / norm.t()
    # K = torch.exp(1.0 * (cosine - 1))  # 2 to increase scale

    dist = utils.batch_pdist(embs, embs)
    K = torch.exp(-10 * dist)
    plt.imshow(K)
    plt.colorbar()
    plt.show()

    Kp = K @ p[0]

    q_logits = (10 + torch.log(p) + torch.randn_like(p)).requires_grad_()
    # q_logits = (10 + torch.log(p)).requires_grad_()
    q_optimizer = optim.Adam([q_logits], lr=5e-3, betas=(0.0, 0.999), amsgrad=False)

    converged = False
    recent = deque(maxlen=5)
    step = 0
    while step < 10000 and not converged:
        q = F.softmax(q_logits, dim=1)
        div = mink_sim_divergence(K, p, q, use_inv=False)
        div_item = div.item()
        if step % 2000 == 0:
            print('Step %d: %0.4f' % (step, div_item))
        recent.append(div_item)
        div.backward()
        q_optimizer.step()
        if np.array(recent).mean() < 0.0001:
            converged = True
        step += 1
    q = q[0]
    Kq = K @ q
    inp = sorted(list(zip(Kq, words)), reverse=True)
    print('\nSummary:', ['%s %0.2f' % (w, p * 100) for (p, w) in inp])
    return Kq, Kp


if __name__ == '__main__':
    with open('data/news_words', 'rb') as f:
        all_words = pickle.load(f)
    with open('data/tfidf', 'rb') as f:
        tdidf_articles = pickle.load(f)
    with open('data/news_vectors', 'rb') as f:
        word_embs = pickle.load(f)
    word_embs = torch.tensor(word_embs, dtype=torch.float32)

    fnames = sorted(glob.glob('data/EnglishProcessed/*.txt'))
    for i, article_fname in enumerate(fnames[5:10], 5):
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
        embs = torch.stack([word_embs[x] for x in feature_index])
        inp = sorted(list(zip(probs, words)), reverse=True)
        print('\nInput:', ['%s %0.2f' % (w, p * 100) for (p, w) in inp])
        print()

        q, Kp = print_summary(words, probs, embs)

        bar_indices = sorted(list(zip(Kp, range(len(feature_index)))), reverse=True)
        bar_data = list(zip(*[(i, p.item() * 100) for (i, (p, w)) in enumerate(bar_indices)]))
        plt.bar(*bar_data, width=1.0, alpha=0.5)

        plot_indices = [q[w] for (_, w) in bar_indices]
        bar_data = list(zip(*[(i, p.item() * 100) for (i, p) in enumerate(plot_indices)]))
        plt.bar(*bar_data, width=1.0, alpha=0.5)
        plt.show()

        # bar_indices = sorted(list(zip(probs, range(len(feature_index)))), reverse=True)
        # bar_data = list(zip(*[(i, p * 100) for (i, (p, w)) in enumerate(bar_indices)]))
        # plt.bar(*bar_data, width=1.0, alpha=0.5)

        # plot_indices = [q[w] for (_, w) in bar_indices]
        # bar_data = list(zip(*[(i, p.item() * 100) for (i, p) in enumerate(plot_indices)]))
        # plt.bar(*bar_data, width=1.0, alpha=0.5)
        # plt.show()

        # bar_indices = sorted(list(zip(Kp, range(len(feature_index)))), reverse=True)
        # plot_indices = [probs[w] for (_, w) in bar_indices]
        # bar_data = list(zip(*[(i, p * 100) for (i, p) in enumerate(plot_indices)]))
        # plt.bar(*bar_data, width=1.0, alpha=0.5)

        # plot_indices = [q[w] for (_, w) in bar_indices]
        # bar_data = list(zip(*[(i, p.item() * 100) for (i, p) in enumerate(plot_indices)]))
        # plt.bar(*bar_data, width=1.0, alpha=0.5)
        # plt.show()

        print("===============")
        print('\n')
