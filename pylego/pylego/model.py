from abc import ABC, abstractmethod
import glob
import pathlib
import sys

import torch
from torch import nn, optim


if sys.version_info.minor < 7:
    class nullcontext():
        def __enter__(self):
            return None
        def __exit__(self, *excinfo):
            pass
else:
    from contextlib import nullcontext


class Model(ABC):

    def __init__(self, model=None, optimizer=None, learning_rate=-1, momentum=-1, cuda=True, load_file=None,
                 save_every=500, save_file=None, max_save_files=5, debug=False):
        self.model = model
        self.save_every = save_every
        self.save_file = save_file
        self.max_save_files = max_save_files
        self.debug = debug
        if debug:
            torch.set_anomaly_enabled(True)

        if isinstance(optimizer, str):
            if optimizer == 'adam':
                if learning_rate < 0.0:
                    learning_rate = 1e-3
                self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            elif optimizer == 'rmspropc':
                if learning_rate < 0.0:
                    learning_rate = 0.01
                if momentum < 0.0:
                    momentum = 0.0
                self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate, momentum=momentum,
                                               centered=True)
        else:
            self.optimizer = optimizer
        self.device = torch.device("cuda" if cuda else "cpu")
        self.train_steps = 0
        if self.model is not None:
            self.model.to(self.device)

        self.initialize(load_file)

    def load(self, load_file):
        """Load a model from a saved file."""
        print("* Loading model from", load_file, "...")
        m_state_dict, o_state_dict, train_steps = torch.load(load_file)
        self.model.load_state_dict(m_state_dict)
        self.optimizer.load_state_dict(o_state_dict)
        self.train_steps = train_steps
        print("* Loaded model from", load_file)

    def initialize(self, load_file):
        """Load a model or initialize it if no valid load file given."""
        if load_file:
            self.load(load_file)

    def save(self, save_file):
        "Save model to file."
        save_fname = save_file + "." + str(self.train_steps)
        print("* Saving model to", save_fname, "...")
        existing = glob.glob(save_file + ".*")
        pairs = [(f.rsplit('.', 1)[-1], f) for f in existing]
        pairs = sorted([(int(k), f) for k, f in pairs if k.isnumeric()], reverse=True)
        for _, fname in pairs[self.max_save_files - 1:]:
            pathlib.Path(fname).unlink()

        save_objs = [self.model.state_dict(), self.optimizer.state_dict(), self.train_steps]
        torch.save(save_objs, save_fname)
        print("* Saved model to", save_fname)

    def set_train(self, train):
        if train:
            self.model.train()
        else:
            self.model.eval()

    def is_training(self):
        return self.model.training

    def get_train_steps(self):
        return self.train_steps

    def prepare_batch(self, data):
        if not isinstance(data, list) and not isinstance(data, tuple):
            data = [data]
        if self.is_training():
            context = nullcontext()
        else:
            context = torch.no_grad()
        with context:
            data = tuple(torch.as_tensor(d, device=self.device) for d in data)
        if len(data) == 1:
            data = data[0]
        return data

    def run_batch(self, data, visualize=False):
        """If visualize is True, a visualize method of the model module is called."""
        if not isinstance(data, list) and not isinstance(data, tuple):
            data = [data]
        if self.is_training():
            context = nullcontext()
        else:
            context = torch.no_grad()
        with context:
            if not visualize:
                return self.model(*data)
            else:
                return self.model.visualize(*data)

    @abstractmethod
    def loss_function(self, forward_ret, labels=None):
        """forward_ret: returned value from run_batch
           labels: additional labels needed to compute loss"""
        pass

    def run_loss(self, data, labels=None):
        """Convenience function to forward the batch and compute the loss."""
        return self.loss_function(self.run_batch(data), labels=labels)

    def increment_train_steps(self):
        self.train_steps += 1
        if (self.save_every > 0 and self.train_steps % self.save_every == 0):
            self.save(self.save_file)

    def train(self, loss, clip_grad_norm=None):
        assert self.is_training()
        self.optimizer.zero_grad()
        loss.backward()
        if clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
        self.optimizer.step()
        self.increment_train_steps()
