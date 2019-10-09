from collections import deque
import glob
import pickle
import sys
import os
sys.path.append('..')

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import argparse
from pylego.misc import add_argument as arg

from renyi import renyi_sim_divergence, rbf_kernel, \
    breg_mixture_divergence_stable, breg_sim_divergence_stable
import utils
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def get_summary_movable_locs(words, probs, embs, all_embs, all_words, k=25, scale=5, autoscale=True):
    """
    This experiment is one where we use a fixed number of (moveable) atoms to approximate the target measure.
    Unfortunately, it doesn't produce particularly nice visualizations, as the moveable atoms must be matched to words
    in our vocabulary, which tends to be a fairly tricky step that doesn't work nearly as well as would be hoped, as the
    "closest" word to each atom is often suprisingly unrelated to the text.
    Instead, see the function after this one, where locations are fixed, which avoids this problem.
    """

    p = torch.tensor(np.array(probs, dtype=np.float32)[None, ...])
    if autoscale:
        dist = utils.batch_pdist(embs, embs, p=2)
        scale = scale*(torch.var(dist))
    kernel = lambda x, y: rbf_kernel(x, y, sigmas=[scale], log=True)
    locs = ((torch.randn_like(embs))[:k] - torch.mean(embs))*torch.std(embs)
    q_logits = torch.zeros(1, k)

    p = p.to(device)
    q_logits = q_logits.to(device).detach().requires_grad_()
    locs = locs.to(device).detach().requires_grad_()
    embs = embs.to(device).detach()
    q_optimizer = optim.Adam([q_logits, locs], lr=1e-4, betas=(0.5, 0.999), amsgrad=False)
    converged = False
    recent = deque(maxlen=1000)
    step = 0
    while step < 100000 and not converged:
        q = F.softmax(q_logits, dim=1)
        div = breg_mixture_divergence_stable(p, embs, q, locs, log_kernel=kernel)

        div_item = div.item()
        loss = div

        if step % 100 == 0:
            big_locs = torch.cat([embs, locs], 0)
            big_q = torch.cat([torch.zeros_like(p), q], 1)
            K = rbf_kernel(big_locs, big_locs, sigmas=[scale])
            print(torch.mean(K).item())
            print(q[0])
            print('Step %d: %0.4f' % (step, div_item))
            inp = sorted(list(zip((K@big_q[0]), words)), reverse=True)
            print('\nSummary:', ['%s %0.2f' % (w, p * 100) for (p, w) in inp])
        recent.append(loss.item())
        loss.backward()
        q_optimizer.step()

        if -np.min(recent) + np.mean(recent) < 1e-4 and step > 1000:
            converged = True
        step += 1

    big_locs = torch.cat([embs, locs], 0)
    big_q = torch.cat([torch.zeros_like(p), q], 1)
    K = rbf_kernel(big_locs, big_locs, sigmas=[scale])
    q = q[0]
    Kq = K @ big_q[0]
    loc_dists = utils.batch_pdist(embs, locs, p=2)

    closest_words = torch.min(loc_dists, 0)[1]
    closest_words = [words[idx] for idx in closest_words]
    dist = utils.batch_pdist(embs, embs, p=2)
    K = torch.exp(-scale*dist**2)
    Kp = K @ p[0]
    inp = sorted(list(zip(probs, words)), reverse=True)
    print('\nInput:', ['%s %0.2f' % (w, p * 100) for (p, w) in inp])
    inp = sorted(list(zip(Kq, words)), reverse=True)
    print('\nSummary:', ['%s %0.2f' % (w, p * 100) for (p, w) in inp])
    inp = sorted(list(zip(q, closest_words)), reverse=True)
    print('\nSummary:', ['%s %0.2f' % (w, p * 100) for (p, w) in inp])

    return Kq, Kp, q, locs, closest_words


def print_summary(words, probs, embs, rbf_sigma=20, rbf=False, cosine_power=1, lda_max=.1, power=.75):
    p = torch.tensor(np.array(probs, dtype=np.float32)[None, ...])

    if rbf:
        dist = utils.batch_pdist(embs, embs, p=2)
        K = torch.exp(-dist**2/rbf_sigma**2)
    else:
        K = (utils.batch_cosine_similarity(embs, embs) + 1)/2
        K = K**cosine_power
    plt.imshow(K)
    plt.colorbar()
    # plt.show()

    Kp = K @ p[0]

    q_logits = torch.zeros_like(p)#(0*torch.randn_like(p)) + torch.log(p)
    #q_logits = q_logits - torch.mean(q_logits)

    p = p.to(device)
    q_logits = q_logits.to(device).detach().requires_grad_()
    K = K.to(device)
    q_optimizer = optim.Adam([q_logits], lr=1e-3, betas=(0.9, 0.999), amsgrad=False)
    converged = False
    recent = deque(maxlen=1000)
    step = 0
    while step < 25000 and not converged:
        lda = lda_max*max(min(step/10000., 1.), 0)
        q = F.softmax(q_logits, dim=1)
        div = breg_sim_divergence_stable(K, p, q)
        reg = lda * torch.norm(q, p=power)
        div_item = div.item()
        loss = div+reg
        loss_item = loss.item()

        if (step - 1) % 100 == 0:
            # print(-np.min(recent) + np.mean(recent), np.min(recent), np.mean(recent))
            print(loss_item, div, reg)

        if step % 1000 == 0:
            print('Step %d: %0.4f; %0.4f' % (step, div_item, loss_item))
            inp = sorted(list(zip(q[0], words)), reverse=True)
            # print('\nSummary:', ['%s %0.2f' % (w, p * 100) for (p, w) in inp])
            inp = sorted(list(zip((K@q[0]), words)), reverse=True)
            # print('\nSummary:', ['%s %0.2f' % (w, p * 100) for (p, w) in inp])
        recent.append(loss.item())
        loss.backward()
        q_optimizer.step()

        if -np.min(recent) + np.mean(recent) < 1e-4 and step > 15000:
            converged = True
        step += 1
    q = q[0]
    Kq = K @ q
    inp = sorted(list(zip(probs, words)), reverse=True)
    # print('\nInput:', ['%s %0.2f' % (w, p * 100) for (p, w) in inp])
    inp = sorted(list(zip(Kq, words)), reverse=True)
    # print('\nSummary:', ['%s %0.2f' % (w, p * 100) for (p, w) in inp])
    inp = sorted(list(zip(q, words)), reverse=True)
    # print('\nSummary:', ['%s %0.2f' % (w, p * 100) for (p, w) in inp])

    return Kq, Kp, q


def wordcloud(probs, words, name, thresh=0.01, hsize=5, vsize=5):
    frequencies = {w.upper():p.item() for w, p in zip(words, probs) if p>thresh}
    cloud = WordCloud(height=int(vsize*400), width=int(hsize*400),
                      background_color="white", colormap="tab20").generate_from_frequencies(frequencies)
    plt.clf()
    plt.figure(figsize=(hsize, vsize))
    plt.imshow(cloud, interpolation='bilinear', aspect="equal")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(name+".pdf", figsize=(5, vsize), dpi=400)

    from scipy.misc import imsave
    imsave(name + '.png', cloud)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg(parser, 'name', type=str, default="", help='name for plots')
    arg(parser, 'raw', type=bool, default=True, help='Use raw vectors rather than UMAP ones.')
    arg(parser, 'alpha', type=float, default=1., help='Float for alpha in divergence.')
    arg(parser, 'rbf', type=bool, default=False, help='Use RBF rather than cosine similarity.')
    arg(parser, 'cosine_power', type=float, default=1, help='Power to raise cosine similarities to.')
    arg(parser, 'power', type=float, default=.75, help='Power for p-norm sparsity constraint.')
    arg(parser, 'rbf_sigma', type=float, default=1, help='Scale for RBF kernel, if used.')
    arg(parser, 'lda', type=float, default=.01, help='Upper limit for sparsity objective.')

    flags = parser.parse_args()
    with open('data/news_words', 'rb') as f:
        all_words = pickle.load(f)
    with open('data/tfidf', 'rb') as f:
        tfidf_articles = pickle.load(f)

    if flags.raw:
        with open('data/news_vectors', 'rb') as f:
            word_embs = pickle.load(f)
    else:
        with open('data/transformed_vectors', 'rb') as f:
            word_embs = pickle.load(f)
    word_embs = torch.tensor(word_embs, dtype=torch.float32)

    if flags.name != "":
        flags.name = flags.name + "/"
        if not os.path.exists("./figs/" + flags.name):
            os.mkdir("./figs/" + flags.name)

    fnames = sorted(glob.glob('data/EnglishProcessed/*.txt'))
    for i, article_fname in enumerate(fnames[6:10], 6):
        print(article_fname)
        feature_index = tfidf_articles[i, :].nonzero()[1]
        if not feature_index.size:
            continue

        with open(article_fname, 'r') as f:
            article = f.read().strip()
        print(article)
        print('-----')
        words = [all_words[x] for x in feature_index]
        print(words)
        probs = [tfidf_articles[i, x] for x in feature_index]
        wordcloud(probs, words, "figs/{}cloud_{}_tfidf".format(flags.name, i-5), 0, hsize=5, vsize=2)
        assert np.isclose(np.sum(probs), 1.0)
        embs = torch.stack([word_embs[x] for x in feature_index])
        inp = sorted(list(zip(probs, words)), reverse=True)
        print('\nInput:', ['%s %0.2f' % (w, p * 100) for (p, w) in inp])
        print()

        Kq, Kp, q = print_summary(words, probs, embs, rbf=flags.rbf, cosine_power=flags.cosine_power, power=flags.power,
                                  rbf_sigma=flags.rbf_sigma, lda_max=flags.lda)

        count = torch.sum(q > 0.01).int().item()
        reduced = inp[:count]
        wordcloud([p for p, w in reduced], [w for p, w in reduced],
                  "figs/{}cloud_{}_tfidf_top_{}".format(flags.name, i-5, count), 0, hsize=2.5, vsize=2)

        wordcloud(q, words, "figs/{}cloud_{}".format(flags.name, i-5), thresh=0.01, hsize=2.5, vsize=2)

        bar_indices = sorted(list(zip(Kp, range(len(feature_index)))), reverse=True)
        bar_data = list(zip(*[(i, p.item() * 100) for (i, (p, w)) in enumerate(bar_indices)]))
        plt.bar(*bar_data, width=1.0, alpha=0.5)

        plot_indices = [q[w] for (_, w) in bar_indices]
        bar_data = list(zip(*[(i, p.item() * 100) for (i, p) in enumerate(plot_indices)]))
        plt.bar(*bar_data, width=1.0, alpha=0.5)
        plt.savefig("KP_plot.pdf")
        # plt.show()

        bar_indices = sorted(list(zip(probs, range(len(feature_index)))), reverse=True)
        bar_data = list(zip(*[(i, p * 100) for (i, (p, w)) in enumerate(bar_indices)]))
        plt.bar(*bar_data, width=1.0, alpha=0.5)

        plot_indices = [q[w] for (_, w) in bar_indices]
        bar_data = list(zip(*[(i, p.item() * 100) for (i, p) in enumerate(plot_indices)]))
        plt.bar(*bar_data, width=1.0, alpha=0.5)
        # plt.show()
        plt.savefig("Probs_plot.pdf")

        bar_indices = sorted(list(zip(Kp, range(len(feature_index)))), reverse=True)
        plot_indices = [probs[w] for (_, w) in bar_indices]
        bar_data = list(zip(*[(i, p * 100) for (i, p) in enumerate(plot_indices)]))
        plt.bar(*bar_data, width=1.0, alpha=0.5)

        plot_indices = [q[w] for (_, w) in bar_indices]
        bar_data = list(zip(*[(i, p.item() * 100) for (i, p) in enumerate(plot_indices)]))
        plt.bar(*bar_data, width=1.0, alpha=0.5)
        # plt.show()
        plt.savefig("KP_plot_2.pdf")

        print("===============")
        print('\n')
