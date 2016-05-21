from os import chdir, path, getcwd

import tarfile

import numpy as np

import theano

from lasagne.layers import DenseLayer, InputLayer, CustomRecurrentLayer
from lasagne.layers.shape import FlattenLayer


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = getcwd()
        chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        chdir(self.savedPath)


def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=path.basename(source_dir))


def split_data(*inputs, split=0.25):
    length = len(inputs[0])
    split_index = int(length * (1 - split))
    outputs = []

    for input in inputs:
        if not len(input) == length:
            raise Exception(
                    'Inputs of different length! (%d and %d)'
                    % (length, len(input)))

        outputs.append(input[:split_index])
        outputs.append(input[split_index:])

    return outputs


def split_test(*inputs, split=0.25):
    return split_data(*inputs, split=split)


def split_val(*inputs, split=0.25):
    return split_data(*inputs, split=split)


def iterate_minibatches(*inputs, batch_size=128, shuffle=True):
    length = len(inputs[0])
    for input in inputs:
        assert len(input) == length

    if shuffle:
        indices = np.arange(length)
        np.random.shuffle(indices)

    for start_index in range(0, length - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_index:start_index + batch_size]
        else:
            excerpt = slice(start_index, start_index + batch_size)

        batch = []
        for input in inputs:
            batch.append(input[excerpt])

        yield batch


def shared_zeros(shape):

    zeros = np.zeros(shape).astype('float32')
    zeros = theano.shared(zeros)

    return zeros


def net_on_seq(net, input_layer):

    net = FlattenLayer(net)

    hidden_shape = net.output_shape
    units = hidden_shape[1]

    l_hid_hid = DenseLayer(
            InputLayer(hidden_shape),
            num_units=units,
            W=shared_zeros((units, units)),
            b=shared_zeros((units,)))
    l_hid_hid.params[l_hid_hid.W].remove('trainable')
    l_hid_hid.params[l_hid_hid.b].remove('trainable')

    net = CustomRecurrentLayer(
            input_layer, net, l_hid_hid)

    return net
