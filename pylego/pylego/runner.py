import collections
import gc
import random
import time
from abc import ABC, abstractmethod

import numpy as np
from tensorboardX import SummaryWriter
import torch


class Runner(ABC):
    """
    Represents a runner that solves a particular task.
    """

    def __init__(self, reader, batch_size, epochs, log_dir, log_keys=None, threads=1, print_every=50,
                 visualize_every=-1, max_batches=-1, seed=0):
        self.reader = reader
        self.batch_size = batch_size
        self.epochs = epochs
        self.print_every = print_every
        self.visualize_every = visualize_every
        self.max_batches = max_batches
        self.model = None
        self.epoch_reports = []
        if log_keys:
            self.log_keys = set(log_keys)
        else:
            self.log_keys = set()
        self.threads = threads

        torch.manual_seed(seed)
        random.seed(seed + 21)
        np.random.seed(seed + 54)
        self.summary_writer = SummaryWriter(log_dir, flush_secs=60)

    @abstractmethod
    def run_batch(self, batch, train=False):
        """Run a session with a batch and return an dict that can be used to print a report.
        It is recommended that stats are averaged over a batch, so that the epoch report is not biased.
        Has to call model.train() for training the model, which is responsible for some bookkeeping."""
        pass

    def report_str(self, report):
        """Format the report in a printable string."""
        ret = []
        for k, v in report.items():
            ret.append('%s: %.3f' % (k, v))
        return ' '.join(ret)

    def print_report(self, epoch, report, step=None):
        """Print the report for a batch if step is given, else print the epoch report."""
        if report:
            report_str = self.report_str(report)
        else:
            report_str = 'nothing to report'
        if step is not None:
            step_str = 'Step %d' % step
        else:
            step_str = 'Summary'
        print('[Epoch %d %s] %s' % (epoch, step_str, report_str))

    def reset_epoch_reports(self):
        """Reset tracked epoch reports for a new epoch."""
        self.epoch_reports = []

    def get_epoch_report(self):
        """Get a consolidated epoch report."""
        n_samples = len(self.epoch_reports)
        if n_samples == 0:
            return None

        keys = list(self.epoch_reports[0].keys())
        epoch_report = collections.OrderedDict([(k, 0.0) for k in keys])
        for report in self.epoch_reports:
            for k, v in report.items():
                epoch_report[k] += v
        for k, v in epoch_report.items():
            epoch_report[k] /= n_samples

        return epoch_report

    def log_report(self, report, train_steps, prefix=""):
        """Log the given report to TensorBoard summary."""
        if report is not None:
            for k, v in report.items():
                if k in self.log_keys:
                    new_key = prefix + k
                    self.summary_writer.add_scalar(new_key, v, global_step=train_steps)

    def log_train_report(self, report, train_steps):
        """Log batch train report."""
        return self.log_report(report, train_steps, prefix="train_")

    def log_epoch_train_report(self, report, train_steps):
        """Log train epoch report."""
        pass

    def log_epoch_val_report(self, report, train_steps):
        """Log val epoch report."""
        return self.log_report(report, train_steps, prefix="val_")

    def train_visualize(self):
        """Visualize during training at regular intervals specified by visualize_every."""
        pass

    def post_epoch_visualize(self, epoch, split):
        """Visualize at the end of an epoch of split.
        epoch=-1 means the model is run only for visualization."""
        pass

    def clean_report(self, ret_report):
        if not ret_report:
            report = collections.OrderedDict()
        elif not isinstance(ret_report, collections.OrderedDict):
            report = collections.OrderedDict()
            for k in sorted(ret_report.keys()):
                report[k] = ret_report[k]
        else:
            report = ret_report
        for k, v in report.items():
            if isinstance(v, torch.Tensor):
                report[k] = v.item()
        return report

    def run_epoch(self, epoch, split, train=False, log=True):
        """Iterates the epoch data for a specific split."""
        print('\n* Starting epoch %d, split %s' % (epoch, split), '(train: ' + str(train) + ')')
        self.reset_epoch_reports()
        self.model.set_train(train)

        timestamp = time.time()
        for i, batch in enumerate(self.reader.iter_batches(split, self.batch_size, shuffle=train,
                                                           partial_batching=not train, threads=self.threads,
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

    def run(self, train_split='train', val_split='val', test_split='test', visualize_only=False,
            visualize_split='test'):
        """Run the main training loop with validation and a final test epoch, or just visualization on the
        test epoch."""
        epoch = -1
        if visualize_only:
            self.model.set_train(False)
            self.post_epoch_visualize(epoch, visualize_split)
        else:
            for epoch in range(self.epochs):
                self.run_epoch(epoch, train_split, train=True)
                if val_split:
                    self.run_epoch(epoch, val_split, train=False)
            if test_split:
                self.run_epoch(epoch + 1, test_split, train=False, log=False)
