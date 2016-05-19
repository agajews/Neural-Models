from os import chdir, path, getcwd

import tarfile

import numpy as np


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
            raise Exception('Inputs of different length! (%d and %d)' % (length, len(input)))

        outputs.append(input[:split_index])
        outputs.append(input[split_index:])

    return outputs


def split_test(*inputs, split=0.25):
    return split_data(*inputs, split=split)


def split_val(*inputs, split=0.25):
    return split_data(*inputs, split=split)


def iterate_minibatches(*inputs, batchsize=128, shuffle=True):
    length = len(inputs[0])
    for input in inputs:
        assert len(input) == length

    if shuffle:
        indices = np.arange(length)
        np.random.shuffle(indices)

    for start_index in range(0, length - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_index:start_index + batchsize]
        else:
            excerpt = slice(start_index, start_index + batchsize)

        batch = []
        for input in inputs:
            batch.append(input[excerpt])

        yield batch
