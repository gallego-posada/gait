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
from wordcloud import WordCloud

from renyi import mink_sim_divergence, renyi_sim_entropy, renyi_sim_divergence
import utils
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

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

    q_logits = (0*torch.randn_like(p))
    # q_logits = (10 + torch.log(p)).requires_grad_()

    p = p.to(device)
    q_logits = q_logits.to(device).detach().requires_grad_()
    K = K.to(device)
    q_optimizer = optim.Adam([q_logits], lr=5e-3, betas=(0.0, 0.999), amsgrad=False)
    converged = False
    recent = deque(maxlen=5)
    step = 0
    while step < 10000 and not converged:
        q = F.softmax(q_logits, dim=1)
        # div = mink_sim_divergence(K, p, q, use_inv=False) + 0.005*renyi_sim_entropy(K, q) #0.005*torch.sum(q * utils.clamp_log(q))
        div = renyi_sim_divergence(K, p, q) + min(step/5000., 1.) * torch.sum(torch.sigmoid((q_logits - torch.mean(q_logits))))
        # div = renyi_sim_divergence(K, p, q) - min(step / 5000., 3.) * torch.clamp(torch.sum(q*utils.clamp_log(q)), 3)
        div_item = div.item()
        if step % 100 == 0:
            print('Step %d: %0.4f' % (step, div_item))
            inp = sorted(list(zip(q[0], words)), reverse=True)
            print('\nSummary:', ['%s %0.2f' % (w, p * 100) for (p, w) in inp])
        recent.append(div_item)
        div.backward()
        q_optimizer.step()
        if np.array(recent).mean() < 0.0001 and step > 100:
            converged = True
        step += 1
    q = q[0]
    Kq = K @ q
    inp = sorted(list(zip(probs, words)), reverse=True)
    print('\nInput:', ['%s %0.2f' % (w, p * 100) for (p, w) in inp])
    inp = sorted(list(zip(Kq, words)), reverse=True)
    print('\nSummary:', ['%s %0.2f' % (w, p * 100) for (p, w) in inp])
    inp = sorted(list(zip(q, words)), reverse=True)
    print('\nSummary:', ['%s %0.2f' % (w, p * 100) for (p, w) in inp])

    return Kq, Kp, q


def wordcloud(probs, words, name, thresh=0.01):
    frequencies = {w.upper():p.item() for w, p in zip(words, probs) if p>thresh}
    cloud = WordCloud(background_color="white", colormap="tab20").generate_from_frequencies(frequencies)

    import matplotlib.pyplot as plt
    plt.clf()
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis("off")
    plt.savefig(name+".pdf")

    # lower max_font_size
    plt.clf()
    cloud = WordCloud(background_color="white", max_font_size=40, colormap="tab20").generate_from_frequencies(frequencies)
    plt.figure()
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    plt.savefig(name+"_small.pdf")


if __name__ == '__main__':
    with open('data/news_words', 'rb') as f:
        all_words = pickle.load(f)
    with open('data/tfidf', 'rb') as f:
        tdidf_articles = pickle.load(f)
    with open('data/transformed_vectors', 'rb') as f:
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

        Kq, Kp, q = print_summary(words, probs, embs)

        wordcloud(Kq, words, "figs/{}_Kq".format(i))
        wordcloud(q, words, "figs/{}_q".format(i))

        bar_indices = sorted(list(zip(Kp, range(len(feature_index)))), reverse=True)
        bar_data = list(zip(*[(i, p.item() * 100) for (i, (p, w)) in enumerate(bar_indices)]))
        plt.bar(*bar_data, width=1.0, alpha=0.5)

        plot_indices = [q[w] for (_, w) in bar_indices]
        bar_data = list(zip(*[(i, p.item() * 100) for (i, p) in enumerate(plot_indices)]))
        plt.bar(*bar_data, width=1.0, alpha=0.5)
        plt.savefig("KP_plot.pdf")
        plt.show()

        bar_indices = sorted(list(zip(probs, range(len(feature_index)))), reverse=True)
        bar_data = list(zip(*[(i, p * 100) for (i, (p, w)) in enumerate(bar_indices)]))
        plt.bar(*bar_data, width=1.0, alpha=0.5)

        plot_indices = [q[w] for (_, w) in bar_indices]
        bar_data = list(zip(*[(i, p.item() * 100) for (i, p) in enumerate(plot_indices)]))
        plt.bar(*bar_data, width=1.0, alpha=0.5)
        plt.show()
        plt.savefig("Probs_plot.pdf")

        bar_indices = sorted(list(zip(Kp, range(len(feature_index)))), reverse=True)
        plot_indices = [probs[w] for (_, w) in bar_indices]
        bar_data = list(zip(*[(i, p * 100) for (i, p) in enumerate(plot_indices)]))
        plt.bar(*bar_data, width=1.0, alpha=0.5)

        plot_indices = [q[w] for (_, w) in bar_indices]
        bar_data = list(zip(*[(i, p.item() * 100) for (i, p) in enumerate(plot_indices)]))
        plt.bar(*bar_data, width=1.0, alpha=0.5)
        plt.show()
        plt.savefig("KP_plot_2.pdf")

        print("===============")
        print('\n')
