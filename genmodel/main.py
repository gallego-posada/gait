import argparse
import glob
import os

from pylego.misc import add_argument as arg

from runners.fixedrunner import FixedRunner
from runners.advrunner import AdversarialRunner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg(parser, 'name', type=str, required=True, help='name of the experiment')
    arg(parser, 'model', type=str, default='fixed.fixed', help='model to use')
    arg(parser, 'cuda', type=bool, default=True, help='enable CUDA')
    arg(parser, 'load_file', type=str, default='', help='file to load model from')
    arg(parser, 'save_file', type=str, default='model.dat', help='model save file')
    arg(parser, 'save_every', type=int, default=300, help='save every these many global steps (-1 to disable saving)')
    arg(parser, 'data_path', type=str, default='data/MNIST')
    arg(parser, 'logs_path', type=str, default='logs')
    arg(parser, 'force_logs', type=bool, default=False)
    arg(parser, 'resume_checkpoint', type=bool, default=True)
    arg(parser, 'learning_rate', type=float, default=1e-3, help='Adam learning rate')
    arg(parser, 'lr_decay', type=float, default=0.99, help='learning rate decay')
    arg(parser, 'beta1', type=float, default=0.9, help='Adam beta1')
    arg(parser, 'beta2', type=float, default=0.999, help='Adam beta2')
    arg(parser, 'grad_norm', type=float, default=5.0, help='gradient norm clipping (-1 to disable)')
    arg(parser, 'batch_size', type=int, default=200, help='batch size')
    arg(parser, 'epochs', type=int, default=350, help='no. of training epochs')
    arg(parser, 'h_size', type=int, default=500, help='hidden state dims')
    arg(parser, 'z_size', type=int, default=2, help='latent dims per layer')
    arg(parser, 'd_size', type=int, default=500, help='disc hidden state dims (-1 for linear disc)')
    arg(parser, 'v_size', type=int, default=16,
        help='dims for vectors to compare in learned costs (-1 for no projection)')
    arg(parser, 'disc_attn', type=bool, default=False, help='weigh input pixels in discriminator')
    arg(parser, 'alpha', type=float, default=1, help='alpha')
    arg(parser, 'use_avg', type=bool, default=False, help='use_avg in renyi mixture divergence')
    arg(parser, 'use_full', type=bool, default=False, help='use_full in renyi mixture divergence')
    arg(parser, 'symmetric', type=bool, default=True, help='symmetric in renyi mixture divergence')
    arg(parser, 'unbiased', type=bool, default=False, help='unbiased gradients mode')
    arg(parser, 'gan_loss', type=str, default='wgan', help='one of: bce, wgan')
    arg(parser, 'kernel', type=str, default='gaussian', help='one of: gaussian, poly, cosine')
    arg(parser, 'kernel_sigma', type=float, default=2.25, help='final sigma for gaussian kernel')
    arg(parser, 'kernel_initial_sigma', type=float, default=2.25, help='initial sigma for gaussian kernel')
    arg(parser, 'sigma_decay_start', type=int, default=-1, help='step to start decaying kernel sigma')
    arg(parser, 'sigma_decay_end', type=int, default=-1, help='step to finish decaying kernel sigma')
    arg(parser, 'kernel_degree', type=float, default=2, help='degree for polynomial or cosine kernel')
    arg(parser, 'sinkhorn_eps', type=float, default=1, help='Sinkhorn epislon')
    arg(parser, 'sinkhorn_iters', type=int, default=10, help='Sinkhorn iters')
    arg(parser, 'gen_iters', type=int, default=3, help='no. of generator iters before discriminator update')
    arg(parser, 'disc_iters', type=int, default=1, help='no. of discriminator iters before generator update')
    arg(parser, 'print_every', type=int, default=50, help='print losses every these many steps')
    arg(parser, 'max_batches', type=int, default=-1, help='max batches per split (for debugging)')
    arg(parser, 'gpus', type=str, default='0')
    arg(parser, 'threads', type=int, default=-1, help='data processing threads (-1 to determine from CPUs)')
    arg(parser, 'debug', type=bool, default=False, help='run model in debug mode')
    arg(parser, 'visualize_every', type=int, default=-1,
        help='visualize during training every these many steps (-1 to disable)')
    arg(parser, 'visualize_only', type=bool, default=False, help='epoch visualize the loaded model and exit')
    arg(parser, 'visualize_split', type=str, default='train', help='split to visualize with visualize_only')
    flags = parser.parse_args()
    if flags.threads < 0:
        flags.threads = max(1, len(os.sched_getaffinity(0)) - 1)
    if flags.grad_norm < 0:
        flags.grad_norm = None

    iters = 0
    while True:
        if iters == 4:
            raise IOError("Too many retries, choose a different name.")
        flags.log_dir = '{}/{}'.format(flags.logs_path, flags.name)
        try:
            print('* Creating log dir', flags.log_dir)
            os.makedirs(flags.log_dir)
            break
        except IOError as e:
            if flags.force_logs:
                print('*', flags.log_dir, 'not recreated')
                break
            else:
                print('*', flags.log_dir, 'already exists')
                flags.name = flags.name + "_"
        iters += 1

    if flags.sigma_decay_end < 0:
        flags.sigma_decay_start = 0
        flags.sigma_decay_end = 1
        flags.kernel_initial_sigma = flags.kernel_sigma

    flags.save_file = flags.log_dir + '/' + flags.save_file
    if flags.resume_checkpoint:
        existing = glob.glob(flags.save_file + ".*")
        pairs = [(f.rsplit('.', 1)[-1], f) for f in existing]
        pairs = sorted([(int(k), f) for k, f in pairs if k.isnumeric()], reverse=True)
        if pairs:
            print('* Checkpoint resuming is enabled, found checkpoint at', pairs[0][1])
            flags.load_file = pairs[0][1]

    print('Arguments:', flags)
    if flags.visualize_only and not flags.load_file:
        print('! WARNING: visualize_only without load_file!')

    if flags.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = flags.gpus

    flags.save_file = flags.log_dir + '/' + flags.save_file

    if flags.model.startswith('fixed.'):
        runner = FixedRunner
    if flags.model.startswith('adv.'):
        runner = AdversarialRunner
    runner(flags).run(val_split=None, test_split=None, visualize_only=flags.visualize_only, visualize_split='train')
