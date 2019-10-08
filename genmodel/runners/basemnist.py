import importlib

from readers import cifar10, fmnist, mnist
from pylego import misc, runner


class MNISTBaseRunner(runner.Runner):

    def __init__(self, flags, model_class, *args, **kwargs):
        self.flags = flags
        if flags.data == 'mnist':
            reader = mnist.MNISTReader(flags.data_path)
        elif flags.data == 'cifar10':
            reader = cifar10.CIFAR10Reader(flags.data_path)
        elif flags.data == 'fmnist':
            reader = fmnist.FashionMNISTReader(flags.data_path)
        summary_dir = flags.log_dir + '/summary'
        super().__init__(reader, flags.batch_size, flags.epochs, summary_dir, threads=flags.threads,
                         print_every=flags.print_every, visualize_every=flags.visualize_every,
                         max_batches=flags.max_batches, *args, **kwargs)
        model_class = misc.get_subclass(importlib.import_module('models.' + self.flags.model), model_class)
        self.model = model_class(self.flags, learning_rate=flags.learning_rate, cuda=flags.cuda,
                                 load_file=flags.load_file, save_every=flags.save_every, save_file=flags.save_file,
                                 debug=flags.debug, max_save_files=5)
