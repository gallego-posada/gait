import gc
import importlib
import time

from readers.mnist import MNISTReader
from pylego import misc, runner


class MNISTBaseRunner(runner.Runner):

    def __init__(self, flags, model_class, log_keys, *args, **kwargs):
        self.flags = flags
        reader = MNISTReader(flags.data_path)
        summary_dir = flags.log_dir + '/summary'
        super().__init__(reader, flags.batch_size, flags.epochs, summary_dir, log_keys=log_keys,
                         threads=flags.threads, print_every=flags.print_every, visualize_every=flags.visualize_every,
                         max_batches=flags.max_batches, *args, **kwargs)
        model_class = misc.get_subclass(importlib.import_module('models.' + self.flags.model), model_class)
        self.model = model_class(self.flags, learning_rate=flags.learning_rate, cuda=flags.cuda,
                                 load_file=flags.load_file, save_every=flags.save_every, save_file=flags.save_file,
                                 debug=flags.debug)

    def run_epoch(self, epoch, split, train=False, log=True):
        """Iterates the epoch data for a specific split."""
        print('\n* Starting epoch %d, split %s' % (epoch, split), '(train: ' + str(train) + ')')
        self.reset_epoch_reports()
        self.model.set_train(train)

        timestamp = time.time()
        for i, batch in enumerate(self.reader.iter_batches(split, self.batch_size, shuffle=train,
                                                           partial_batching=False, threads=self.threads,
                                                           max_batches=self.max_batches)):

            report = self.clean_report(self.run_batch(batch, train=train))
            report['time_'] = time.time() - timestamp
            if train:
                self.log_train_report(report, self.model.get_train_steps())
            self.epoch_reports.append(report)
            if self.print_every > 0 and i % self.print_every == 0:
                self.print_report(epoch, report, step=i)
            if train and self.visualize_every > 0 and self.model.get_train_steps() % self.visualize_every == 0:
                self.model.set_train(False)
                self.train_visualize()
                self.model.set_train(True)

            timestamp = time.time()

        epoch_report = self.get_epoch_report()
        self.print_report(epoch, epoch_report)
        if train:
            self.log_epoch_train_report(epoch_report, self.model.get_train_steps())
        elif log:
            self.log_epoch_val_report(epoch_report, self.model.get_train_steps())

        self.model.set_train(False)
        self.post_epoch_visualize(epoch, split)
        gc.collect()
