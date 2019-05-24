import numpy as np
from PIL import Image


def print_gradient(name):
    '''To be used with .register_hook'''
    return lambda grad: print(name, grad.norm(p=2).item())


def add_argument(parser, flag, type=None, **kwargs):
    """Wrapper to add arguments to an argument parser. Fixes argparse's
    behavior with type=bool. For a bool flag 'test', this adds options '--test'
    which by default sets test to on, and additionally supports '--test true',
    '--test false' and so on. Finally, 'test' can also be turned off by simply
    specifying '--notest'.
    """
    def str2bool(v):
        return v.lower() in ('true', 't', '1')
    if flag.startswith('-'):
        raise ValueError('Flags should not have the preceeding - symbols, -- will be added automatically.')
    if type == bool:
        parser.add_argument('--' + flag, type=str2bool, nargs='?', const=True, **kwargs)
        parser.add_argument('--no' + flag, action='store_false', dest=flag)
    else:
        parser.add_argument('--' + flag, type=type, **kwargs)


def get_subclass(module, base_class):
    ret = []
    for name in dir(module):
        obj = getattr(module, name)
        try:
            if issubclass(obj, base_class) and obj is not base_class:
                ret.append(obj)
                if len(ret) > 1:
                    raise ValueError("Module " + module.__name__ + " has more than one class subclassing " +
                                     base_class.__name__)
        except TypeError:  # 'obj' is not a class
            pass
    if len(ret) == 0:
        raise ValueError("Module " + module.__name__ + " doesn't have a class subclassing " + base_class.__name__)
    return ret[0]


def nan_check(debug_str, variable):
    print(debug_str, np.any(np.isnan(variable)))


def scaled_int(tensor, scale=1.0):
    return (tensor * float(scale)).astype(np.int32)


def save_comparison_grid(fname, *args, border_width=2, border_shade=0.0, desired_aspect=1.0, format='nchw',
                         rows_cols=None, retain_sequence=False):
    """Arrange image batches in a grid such that corresponding images in *args are next to each other.
    All images should be in range [0,1].

    The automatic behavior can be overridden by providing the following arguments:
        rows_cols: tuple (n_rows, n_cols)
        retain_sequence: retain original image sequence if True
    """
    assert np.all(args[0].shape == arg.shape for arg in args)
    args = np.array(args)
    if format == 'nchw':
        args = np.transpose(args, (0, 1, 3, 4, 2))
    else:
        assert format == 'nhwc'

    args = np.concatenate([args, border_shade * np.ones([args.shape[0], args.shape[1], border_width, args.shape[3],
                                                         args.shape[4]])], axis=2)
    args = np.concatenate([args, border_shade * np.ones([args.shape[0], args.shape[1], args.shape[2], border_width,
                                                         args.shape[4]])], axis=3)
    args = np.concatenate(args, axis=2)

    if rows_cols is None:
        aspect_ratio = args.shape[2] / args.shape[1]
        scale_aspect = aspect_ratio / desired_aspect

        # we want to divide width by scale_aspect, or multiply height by it
        # want nH * nW = N, with nH / nW = S => nH = S * nW
        # nW = sqrt(N/S), nH = S*nW
        nW = np.sqrt(args.shape[0] / scale_aspect)
        nH = scale_aspect * nW

        w_aspect = (np.ceil(nW) * args.shape[2]) / (np.floor(nH) * args.shape[1])
        h_aspect = (np.floor(nW) * args.shape[2]) / (np.ceil(nH) * args.shape[1])
        wh_aspect = (np.ceil(nW) * args.shape[2]) / (np.ceil(nH) * args.shape[1])
        w_diff = (np.abs(w_aspect - desired_aspect), (np.floor(nH), np.ceil(nW)))
        h_diff = (np.abs(h_aspect - desired_aspect), (np.ceil(nH), np.floor(nW)))
        wh_diff = (np.abs(wh_aspect - desired_aspect), (np.ceil(nH), np.ceil(nW)))

        for _, (h, w) in sorted([w_diff, h_diff, wh_diff]):
            if h * w >= args.shape[0]:
                nH = h
                nW = w
                break
        nH, nW = int(nH), int(nW)
    else:
        nH, nW = rows_cols

    if retain_sequence or args.shape[1] <= args.shape[2]:  # keep space at bottom
        while True:
            args_block = args[:(nH - 1) * nW]
            if args_block.shape[0] < (nH - 1) * nW:
                nH -= 1
            else:
                break
        args_block = args_block.reshape(nH - 1, nW, args_block.shape[1], args_block.shape[2], args_block.shape[3])
        args_bottom = args[(nH - 1) * nW:]
        if args_bottom.shape[0] > 0:
            args_bottom = np.concatenate([args_bottom,
                                          border_shade * np.ones([args_block.shape[1] - args_bottom.shape[0],
                                                                  args_bottom.shape[1], args_bottom.shape[2],
                                                                  args_bottom.shape[3]])], axis=0)
            args_bottom = args_bottom[None, ...]
            args = np.concatenate([args_block, args_bottom], axis=0)
        else:
            args = args_block
    else:  # keep space at right
        while True:
            args_block = args[:nH * (nW - 1)]
            if args_block.shape[0] < nH * (nW - 1):
                nW -= 1
            else:
                break
        args_block = args_block.reshape(nH, nW - 1, args_block.shape[1], args_block.shape[2], args_block.shape[3])
        args_right = args[nH * (nW - 1):]
        if args_right.shape[0] > 0:
            args_right = np.concatenate([args_right, border_shade * np.ones([args_block.shape[0] - args_right.shape[0],
                                                                             args_right.shape[1], args_right.shape[2],
                                                                             args_right.shape[3]])], axis=0)
            args_right = args_right[:, None, ...]
            args = np.concatenate([args_block, args_right], axis=1)
        else:
            args = args_block

    args = np.transpose(args, (0, 2, 1, 3, 4))
    args = args.reshape(args.shape[0] * args.shape[1], args.shape[2] * args.shape[3], args.shape[4])
    args = np.concatenate([border_shade * np.ones([border_width, args.shape[1], args.shape[2]]), args], axis=0)
    args = np.concatenate([border_shade * np.ones([args.shape[0], border_width, args.shape[2]]), args], axis=1)

    if args.shape[-1] == 1:
        args = args[:, :, 0]

    im = Image.fromarray((args * 255).astype(np.uint8))
    im.save(fname)


class LinearDecay:
    '''Utility class for linear decay that provides a method to return the current decay value at given x.'''

    def __init__(self, x_start, x_end, y_start, y_end, bound_y=True):
        '''Initialize using x and y limits. If bound_y is True, y is always bounded between y_start and y_end.'''
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.bound_y = bound_y

    def get_y(self, x):
        y = (((x - self.x_start) * self.y_end) + ((self.x_end - x) * self.y_start)) / (self.x_end - self.x_start)
        if self.bound_y:
            if self.y_end > self.y_start:
                y = max(min(y, self.y_end), self.y_start)
            else:
                y = max(min(y, self.y_start), self.y_end)
        return y


# SumTree based on the original implementation at https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
class SumTree:
    '''SumTree: a binary tree data structure where the parent’s value is the sum of its children'''

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.count = 0

    def _propagate(self, idx, change):
        '''update to the root node'''
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        '''find sample on leaf node'''
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        '''sum of all keys'''
        return self.tree[0]

    def add(self, p, data):
        '''store priority and sample'''
        idx = self.write + self.capacity - 1
        old_obj = self.data[self.write]
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.count < self.capacity:
            self.count += 1
        return old_obj

    def update(self, idx, p):
        '''update priority'''
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        '''get priority and sample'''
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])
