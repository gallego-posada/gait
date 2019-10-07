import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

LOG2PI = np.log(2.0 * np.pi)


class View(nn.Module):

    def __init__(self, *view_as):
        super().__init__()
        self.view_as = view_as

    def forward(self, x):
        return x.view(*self.view_as)


class Upsample(nn.Module):

    def __init__(self, scale_factor=2, mode='bilinear'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)


class InstanceNormFC(nn.Module):

    def __init__(self, _unused=0, affine=True):
        super().__init__()
        self.norm = nn.InstanceNorm1d(1, affine=affine)

    def forward(self, x):
        return self.norm(x.unsqueeze(1)).squeeze(1)


class GridGaussian(nn.Module):
    '''Projects input coordinates [y, x] to a grid of size [h, w] with a 2D Gaussian of mean [y, x] and std sigma.'''

    def __init__(self, variance, h, w, hmin, hmax, wmin, wmax, mean_value=None):
        super().__init__()
        self.variance = variance
        self.h = h
        self.w = w
        if mean_value is None:
            self.mean_value = 1.0 / (2.0 * np.pi * variance)  # From pdf of Gaussian
        else:
            self.mean_value = mean_value
        ones = np.ones([h, w])
        ys_channel = np.linspace(hmin, hmax, h)[:, np.newaxis] * ones
        xs_channel = np.linspace(wmin, wmax, w)[np.newaxis, :] * ones
        initial = np.concatenate([ys_channel[np.newaxis, ...], xs_channel[np.newaxis, ...]], 0)  # 2 x h x w
        self.linear_grid = nn.Parameter(torch.Tensor(initial), requires_grad=False)

    def forward(self, loc):
        '''loc has shape [..., 2], where loc[...] = [y_i x_i].'''
        loc_grid = loc[..., None, None].expand(*loc.size(), self.h, self.w)
        expanded_lin_grid = self.linear_grid[None, ...].expand_as(loc_grid)
        # both B x 2 x h x w
        reduction_dim = len(loc_grid.size()) - 3
        return ((-(expanded_lin_grid - loc_grid).pow(2).sum(dim=reduction_dim) / (2.0 * self.variance)).exp() *
                self.mean_value)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, rescale=None, norm=None, nonlinearity=F.elu, final=False,
                 skip_last_norm=False, fixup_l=1, negative_slope=0.0, enable_gain=True, cond_model=None):
        super().__init__()
        self.final = final
        self.skip_last_norm = skip_last_norm
        if stride < 0:
            self.upsample = Upsample(-stride)
            stride = 1
        else:
            self.upsample = nn.Identity()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if norm is not None:
            self.bn1 = norm(planes, affine=cond_model is None)
        else:
            self.bn1 = nn.Identity()
        self.nonlinearity = nonlinearity
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        if norm is not None:
            self.bn2 = norm(planes, affine=cond_model is None)
        else:
            self.bn2 = nn.Identity()
        if cond_model is not None:
            self.cond1 = cond_model(planes)
            self.cond2 = cond_model(planes)
        else:
            self.cond1, self.cond2 = None, None
        self.rescale = rescale
        self.stride = stride
        if not enable_gain:
            self.gain = 1.0  # disable gain if we're trying to Lipschitz constrain the module
        else:
            self.gain = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, 1, 1)) for _ in range(4)])

        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', a=negative_slope)
        self.conv1.weight.data.mul_(fixup_l ** (-0.5))
        if not enable_gain:
            nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', a=negative_slope)
            self.conv2.weight.data.mul_(fixup_l ** (-0.5))
        else:
            self.conv2.weight.data.zero_()

    def forward(self, args):
        x, y = args
        identity = x

        out = self.upsample(x + self.biases[0])
        out = self.conv1(out) + self.biases[1]
        out = self.bn1(out)
        if self.cond1 is not None:
            w, b = self.cond1(y)
            out = out * w[:, :, None, None] + b[:, :, None, None]
        out = self.nonlinearity(out) + self.biases[2]

        out = self.gain * self.conv2(out) + self.biases[3]
        if not self.final or not self.skip_last_norm:
            out = self.bn2(out)
        if self.cond2 is not None:
            w, b = self.cond2(y)
            out = out * w[:, :, None, None] + b[:, :, None, None]

        if self.rescale is not None:
            rescale = self.rescale
            if self.final and self.skip_last_norm:
                rescale = rescale[:-1]
            identity = rescale(x + self.biases[0])

        out += identity
        if not self.final:
            out = self.nonlinearity(out)

        return out, y


class ResNet(nn.Module):

    def __init__(self, inplanes, layers, block=None, norm=None, nonlinearity=F.elu, skip_last_norm=False,
                 total_layers=-1, negative_slope=0.0, enable_gain=True, cond_model=None):
        '''layers is a list of tuples (layer_size, input_planes, stride). Negative stride for upscaling.
        If cond_model is not None, it should be a function that can create a module based on constructor arg 'channels'.
        The module should take as input a conditioning variable and producing two outputs (weight, bias) of sizes
        (batch_size, channels or 1).'''
        super().__init__()
        self.norm = norm
        self.skip_last_norm = skip_last_norm
        self.negative_slope = negative_slope
        self.enable_gain = enable_gain
        self.cond_model = cond_model
        if block is None:
            block = ResBlock

        self.inplanes = inplanes
        self.nonlinearity = nonlinearity
        all_layers = []
        if total_layers < 0:
            fixup_l = sum(l[0] for l in layers)
        else:
            fixup_l = total_layers
        for i, (layer_size, inplanes, stride) in enumerate(layers):
            final = (i == len(layers) - 1)
            all_layers.append(self._make_layer(block, inplanes, layer_size, stride=stride, final=final,
                                               fixup_l=fixup_l))
        self.layers = nn.Sequential(*all_layers)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, final=False, fixup_l=1):
        rescale = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.norm is not None:
                batch_norm2d = self.norm(planes * block.expansion, affine=True)
            else:
                batch_norm2d = nn.Identity()
            if stride < 0:
                stride_ = -stride
                rescale = nn.Sequential(
                    Upsample(stride_),
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, bias=False),
                    batch_norm2d,
                )
                conv = 1
            else:
                rescale = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    batch_norm2d,
                )
                conv = 0
            nn.init.kaiming_normal_(rescale[conv].weight, mode='fan_out', a=self.negative_slope)

        layers = []
        layer_final = final and blocks == 1
        layers.append(block(self.inplanes, planes, stride, rescale, norm=self.norm, nonlinearity=self.nonlinearity,
                            final=layer_final, skip_last_norm=self.skip_last_norm, fixup_l=fixup_l,
                            negative_slope=self.negative_slope, enable_gain=self.enable_gain,
                            cond_model=self.cond_model))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layer_final = final and i == blocks - 1
            layers.append(block(self.inplanes, planes, norm=self.norm, nonlinearity=self.nonlinearity,
                                final=layer_final, skip_last_norm=self.skip_last_norm, fixup_l=fixup_l,
                                negative_slope=self.negative_slope, enable_gain=self.enable_gain,
                                cond_model=self.cond_model))

        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        return self.layers((x, y))[0]


class ResBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, hidden_size, rescale=None, norm=None, nonlinearity=F.elu, final=False,
                 skip_last_norm=False, fixup_l=1, enable_gain=True, cond_model=None):
        super().__init__()
        self.final = final
        self.skip_last_norm = skip_last_norm
        self.fc1 = nn.Linear(inplanes, hidden_size, bias=False)
        if norm is not None:
            if norm == nn.LayerNorm:
                self.bn1 = norm(hidden_size, elementwise_affine=cond_model is None)
            else:
                self.bn1 = norm(hidden_size, affine=cond_model is None)
        else:
            self.bn1 = nn.Identity()
        self.nonlinearity = nonlinearity
        self.fc2 = nn.Linear(hidden_size, planes, bias=False)
        if norm is not None:
            if norm == nn.LayerNorm:
                self.bn2 = norm(planes, elementwise_affine=cond_model is None)
            else:
                self.bn2 = norm(planes, affine=cond_model is None)
        else:
            self.bn2 = nn.Identity()
        if cond_model is not None:
            self.cond1 = cond_model(hidden_size)
            self.cond2 = cond_model(planes)
        else:
            self.cond1, self.cond2 = None, None
        self.rescale = rescale
        if not enable_gain:
            self.gain = 1.0  # disable gain if we're trying to Lipschitz constrain the module
        else:
            self.gain = nn.Parameter(torch.ones(1, 1))
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(1, 1)) for _ in range(4)])

        nn.init.xavier_normal_(self.fc1.weight)
        self.fc1.weight.data.mul_(fixup_l ** (-0.5))
        if not enable_gain:
            nn.init.xavier_normal_(self.fc2.weight)
            self.fc2.weight.data.mul_(fixup_l ** (-0.5))
        else:
            self.fc2.weight.data.zero_()

    def forward(self, args):
        x, y = args
        identity = x

        out = x + self.biases[0]
        out = self.fc1(out) + self.biases[1]
        out = self.bn1(out)
        if self.cond1 is not None:
            w, b = self.cond1(y)
            out = out * w + b
        out = self.nonlinearity(out) + self.biases[2]

        out = self.gain * self.fc2(out) + self.biases[3]
        if not self.final or not self.skip_last_norm:
            out = self.bn2(out)
        if self.cond2 is not None:
            w, b = self.cond2(y)
            out = out * w + b

        if self.rescale is not None:
            rescale = self.rescale
            if self.final and self.skip_last_norm:
                rescale = rescale[:-1]
            identity = rescale(x + self.biases[0])

        out += identity
        if not self.final:
            out = self.nonlinearity(out)

        return out, y


class ResNet1d(nn.Module):

    def __init__(self, inplanes, layers, block=None, norm=None, nonlinearity=F.elu, skip_last_norm=False,
                 total_layers=-1, enable_gain=True, cond_model=None):
        '''layers is a list of tuples (layer_size, inout_planes, hidden_size).
        If cond_model is not None, it should be a function that can create a module based on constructor arg 'dims'.
        The module should take as input a conditioning variable and producing two outputs (weight, bias) of sizes
        (batch_size, dims or 1).'''
        super().__init__()
        self.norm = norm
        self.skip_last_norm = skip_last_norm
        self.enable_gain = enable_gain
        self.cond_model = cond_model
        if block is None:
            block = ResBlock1d

        self.inplanes = inplanes
        self.nonlinearity = nonlinearity
        all_layers = []
        if total_layers < 0:
            fixup_l = sum(l[0] for l in layers)
        else:
            fixup_l = total_layers
        for i, (layer_size, inplanes, hidden_size) in enumerate(layers):
            final = (i == len(layers) - 1)
            all_layers.append(self._make_layer(block, inplanes, hidden_size, layer_size, final=final, fixup_l=fixup_l))
        self.layers = nn.Sequential(*all_layers)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.InstanceNorm1d) or isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, hidden_size, blocks, final=False, fixup_l=1):
        rescale = None
        if self.inplanes != planes * block.expansion:
            if self.norm is not None:
                if self.norm == nn.LayerNorm:
                    batch_norm1d = self.norm(planes * block.expansion, elementwise_affine=True)
                else:
                    batch_norm1d = self.norm(planes * block.expansion, affine=True)
            else:
                batch_norm1d = nn.Identity()
            rescale = nn.Sequential(
                nn.Linear(self.inplanes, planes * block.expansion, bias=False),
                batch_norm1d,
            )
            nn.init.xavier_normal_(rescale[0].weight)

        layers = []
        layer_final = final and blocks == 1
        layers.append(block(self.inplanes, planes, hidden_size, rescale=rescale, norm=self.norm,
                            nonlinearity=self.nonlinearity, final=layer_final, skip_last_norm=self.skip_last_norm,
                            fixup_l=fixup_l, enable_gain=self.enable_gain, cond_model=self.cond_model))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layer_final = final and i == blocks - 1
            layers.append(block(self.inplanes, planes, hidden_size, norm=self.norm, nonlinearity=self.nonlinearity,
                                final=layer_final, skip_last_norm=self.skip_last_norm, fixup_l=fixup_l,
                                enable_gain=self.enable_gain, cond_model=self.cond_model))

        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        return self.layers((x, y))[0]


class MultilayerLSTMCell(nn.Module):
    '''Provides a mutli-layer wrapper for LSTMCell.'''

    def __init__(self, input_size, hidden_size, bias=True, layers=1, every_layer_input=False,
                 use_previous_higher=False):
        '''
        every_layer_input: Consider raw input at every layer.
        use_previous_higher: Take higher layer at previous timestep as input to current layer.
        '''
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.every_layer_input = every_layer_input
        self.use_previous_higher = use_previous_higher
        input_sizes = [input_size] + [hidden_size for _ in range(1, layers)]
        if every_layer_input:
            for i in range(1, layers):
                input_sizes[i] += input_size
        if use_previous_higher:
            for i in range(layers - 1):
                input_sizes[i] += hidden_size
        self.lstm_cells = nn.ModuleList([nn.LSTMCell(input_sizes[i], hidden_size, bias=bias) for i in range(layers)])

    def forward(self, input_, hx=None):
        '''
        Input: input, [(h_0, c_0), ..., (h_L, c_L)]
        Output: [(h_0, c_0), ..., (h_L, c_L)]
        '''
        if hx is None:
            hx = [None] * self.layers
        outputs = []
        recent = input_
        for layer in range(self.layers):
            if layer > 0 and self.every_layer_input:
                recent = torch.cat([recent, input_], dim=1)
            if layer < self.layers - 1 and self.use_previous_higher:
                if hx[layer + 1] is None:
                    prev = recent.new_zeros([recent.size(0), self.hidden_size])
                else:
                    prev = hx[layer + 1][0]
                recent = torch.cat([recent, prev], dim=1)
            out = self.lstm_cells[layer](recent, hx[layer])
            recent = out[0]
            outputs.append(out)
        return outputs


class MultilayerLSTM(nn.Module):
    '''A multilayer LSTM that uses MultilayerLSTMCell.'''

    def __init__(self, input_size, hidden_size, bias=True, layers=1, every_layer_input=False,
                 use_previous_higher=False):
        super().__init__()
        self.cell = MultilayerLSTMCell(input_size, hidden_size, bias=bias, layers=layers,
                                       every_layer_input=every_layer_input, use_previous_higher=use_previous_higher)

    def forward(self, input_, reset=None):
        '''If reset is 1.0, the RNN state is reset AFTER that timestep's output is produced, otherwise if reset is 0.0,
        nothing is changed.'''
        hx = None
        outputs = []
        for t in range(input_.size(1)):
            hx = self.cell(input_[:, t], hx)
            outputs.append(torch.cat([h[:, None, None, :] for (h, c) in hx], dim=2))
            if reset is not None:
                reset_t = reset[:, t, None]
                if torch.any(reset_t > 1e-6):
                    for i, (h, c) in enumerate(hx):
                        hx[i] = (h * (1.0 - reset_t), c * (1.0 - reset_t))

        return torch.cat(outputs, dim=1)  # size: batch_size, length, layers, hidden_size


class ConvLSTMCell(nn.Module):
    """
    Basic CLSTM cell.

    Code borrowed with thanks from:
    Shreyas Padhy, Andrea Palazzi, and Sreenivas Venkobarao

    https://github.com/ndrplz/ConvLSTM_pytorch
    https://github.com/shreyaspadhy/UNet-Zoo/blob/master/CLSTM.py
    https://github.com/SreenivasVRao/ConvGRU-ConvLSTM-PyTorch/blob/master/convlstm.py
    """

    def __init__(self, in_channels, hidden_channels, kernel_size, bias):

        super().__init__()

        self.input_dim = in_channels
        self.hidden_dim = hidden_channels

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # TODO allow using a small ResNet
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, b, h, w, dtype, device):
        return (torch.zeros(b, self.hidden_dim, h, w, dtype=dtype, device=device),
                torch.zeros(b, self.hidden_dim, h, w, dtype=dtype, device=device))


class ConvLSTM(nn.Module):  # TODO use_previous_higher, every_layer_input
    """
    Convolutional LSTM

    Code borrowed with thanks from:
    Shreyas Padhy, Andrea Palazzi, and Sreenivas Venkobarao

    https://github.com/ndrplz/ConvLSTM_pytorch
    https://github.com/shreyaspadhy/UNet-Zoo/blob/master/CLSTM.py
    https://github.com/SreenivasVRao/ConvGRU-ConvLSTM-PyTorch/blob/master/convlstm.py
    """

    def __init__(self, in_channels, hidden_channels, kernel_size, num_layers, bias=True, return_all_layers=False):
        '''Input and output are (batch, time, channels, height, width)'''
        super().__init__()
        if not isinstance(kernel_size, tuple):
            kernel_size = (kernel_size, kernel_size)
        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_channels` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_channels = self._extend_for_multilayer(hidden_channels, num_layers)
        if not len(kernel_size) == len(hidden_channels) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = in_channels
        self.hidden_dim = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]

            cell_list.append(ConvLSTMCell(in_channels=cur_input_dim,
                                          hidden_channels=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None, reset=None):
        '''If reset is 1.0, the RNN state is reset AFTER that timestep's output is produced, otherwise if reset is 0.0,
        nothing is changed.'''

        if reset is not None and not torch.any(reset > 1e-6):
            reset = None
        if reset is not None:
            reset = reset[..., None, None, None]

        # TODO implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            b, _, _, h, w = input_tensor.shape
            hidden_state = self._init_hidden(b, h, w, input_tensor.dtype, input_tensor.device)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        if reset is not None:
            resets = [reset[:, t, :, :, :] for t in range(seq_len - 1)]

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                if t > 0 and reset is not None:
                    reset_t = resets[t - 1]
                    h = h * (1.0 - reset_t)
                    c = c * (1.0 - reset_t)
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return torch.cat(layer_output_list, dim=2), last_state_list

    def _init_hidden(self, b, h, w, dtype, device):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(b, h, w, dtype, device))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvBLSTM(nn.Module):
    """
    Convolutional BiLSTM

    Code borrowed with thanks from:
    Shreyas Padhy, Andrea Palazzi, and Sreenivas Venkobarao

    https://github.com/ndrplz/ConvLSTM_pytorch
    https://github.com/shreyaspadhy/UNet-Zoo/blob/master/CLSTM.py
    https://github.com/SreenivasVRao/ConvGRU-ConvLSTM-PyTorch/blob/master/convlstm.py
    """

    def __init__(self, in_channels, hidden_channels, kernel_size, num_layers, bias=True):

        super().__init__()
        self.forward_net = ConvLSTM(in_channels, hidden_channels//2, kernel_size, num_layers, bias=bias)
        self.reverse_net = ConvLSTM(in_channels, hidden_channels//2, kernel_size, num_layers, bias=bias)

    def forward(self, xforward, xreverse):
        """
        xforward, xreverse = B T C H W tensors.
        """

        y_out_fwd, _ = self.forward_net(xforward)
        y_out_rev, _ = self.reverse_net(xreverse)

        y_out_fwd = y_out_fwd[-1]  # outputs of last CLSTM layer = B, T, C, H, W
        y_out_rev = y_out_rev[-1]  # outputs of last CLSTM layer = B, T, C, H, W

        reversed_idx = list(reversed(range(y_out_rev.shape[1])))
        y_out_rev = y_out_rev[:, reversed_idx, ...]  # reverse temporal outputs.
        ycat = torch.cat((y_out_fwd, y_out_rev), dim=2)

        return ycat


class SpectralNorm1d(object):
    _version = 1

    def __init__(self, name='weight', eps=1e-6):
        self.name = name
        self.eps = eps

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        sigma = torch.clamp(weight.abs().max(), min=self.eps)
        return weight / sigma

    def remove(self, module):
        with torch.no_grad():
            weight = self.compute_weight(module)
        delattr(module, self.name)
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))

    @staticmethod
    def apply(module, name, eps):
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, SpectralNorm1d) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm1d hooks on the same parameter {}".format(name))

        fn = SpectralNorm1d(name, eps)
        weight = module._parameters[name]

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, fn.name, weight.data)

        module.register_forward_pre_hook(fn)
        return fn


def spectral_norm1d(module, name='weight', eps=1e-6):
    SpectralNorm1d.apply(module, name, eps)
    return module


def remove_spectral_norm1d(module, name='weight'):
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm1d) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module
    raise ValueError("spectral_norm1d of '{}' not found in {}".format(name, module))


def thresholded_sigmoid(x, linear_range=0.8):
    # t(x)={-l<=x<=l:0.5+x, x<-l:s(x+l)(1-2l), x>l:s(x-l)(1-2l)+2l}
    l = linear_range / 2.0
    return torch.where(x < -l, torch.sigmoid(x + l) * (1. - linear_range),
                       torch.where(x > l, torch.sigmoid(x - l) * (1. - linear_range) + linear_range, x + 0.5))


def inv_thresholded_sigmoid(x, linear_range=0.8):
    # t^-1(x)={0.5-l<=x<=0.5+l:x-0.5, x<0.5-l:-l-ln((1-2l-x)/x), x>0.5+l:l-ln((1-x)/(x-2l))}
    l = linear_range / 2.0
    return torch.where(x < 0.5 - l, -l - torch.log((1. - linear_range - x) / x),
                       torch.where(x > 0.5 + l, l - torch.log((1. - x) / (x - linear_range)), x - 0.5))


def reparameterize_gaussian(mu, logvar, sample, return_eps=False):
    std = torch.exp(0.5 * logvar)
    if sample:
        eps = torch.randn_like(std)
    else:
        eps = torch.zeros_like(std)
    ret = eps.mul(std).add_(mu)
    if return_eps:
        return ret, eps
    else:
        return ret


def kl_div_gaussian(q_mu, q_logvar, p_mu=None, p_logvar=None):
    '''Batched KL divergence D(q||p) computation.'''
    if p_mu is None or p_logvar is None:
        zero = q_mu.new_zeros(1)
        p_mu = p_mu or zero
        p_logvar = p_logvar or zero
    logvar_diff = q_logvar - p_logvar
    kl_div = -0.5 * (1.0 + logvar_diff - logvar_diff.exp() - ((q_mu - p_mu)**2 / p_logvar.exp()))
    return kl_div.sum(dim=-1)


def gaussian_log_prob(mu, logvar, x):
    '''Batched log probability log p(x) computation.'''
    logprob = -0.5 * (LOG2PI + logvar + ((x - mu)**2 / logvar.exp()))
    return logprob.sum(dim=-1)
