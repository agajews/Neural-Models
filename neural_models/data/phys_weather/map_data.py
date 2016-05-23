from os.path import isfile

from cv2 import imread, resize, cvtColor, COLOR_BGR2HSV

import pickle

import numpy as np

from neural_models.lib import split_test
from .station_data import get_days_list, display_station_data, map_fnm_fn, gen_station_np_seq


def get_channels(color):

    if color == 'rgb':
        channels = 3

    elif color == 'hsv':
        channels = 1

    else:
        raise Exception('Invalid color %s' % color)

    return channels


def proc_rgb_image(image, channels, width, height):

    image_np = np.zeros((channels, width, height))
    image = np.transpose(image, (2, 0, 1))

    for channel in range(channels):
        image = resize(image[channel, :, :], (height, width))
        image_np[channel, :, :] = image

    return image_np


def proc_hsv_image(image, width, height):

    image_np = np.zeros((1, width, height))

    image = cvtColor(image, COLOR_BGR2HSV)
    image = resize(image[:, :, 0], (height, width))

    image_np[:, :, :] = image

    return image_np


def load_image_np(item_list, fnm_fn, width, height, color):

    channels = get_channels(color)

    image_np = np.zeros((len(item_list), channels, width, height))

    for i, item in enumerate(item_list):
        image_fnm = fnm_fn(item)
        image = imread(image_fnm)

        if color == 'rgb':
            proc_image = proc_rgb_image(image, channels, width, height)

        elif color == 'hsv':
            proc_image = proc_hsv_image(image, width, height)

        image_np[i, :, :, :] = proc_image

    return image_np


def get_image_meta(image_np):

    num_days, channels, _, _ = image_np.shape
    return num_days, channels


def gen_image_np_seq(image_np, width, height, timesteps):

    num_days, channels = get_image_meta(image_np)

    X = np.zeros((num_days - timesteps, timesteps, channels, width, height))

    for day_num in range(timesteps, num_days):
        example_num = day_num - timesteps

        for hist_num in range(timesteps):
            X[example_num, hist_num, :, :, :] = \
                image_np[example_num + hist_num, :, :, :]

    return X


def gen_temp_data(width, height, timesteps,
        verbose, color, elem):

    dly_fnm = 'raw_data/phys_weather/chicago_summaries.dly'
    days_list = get_days_list(dly_fnm, map_filter=True)

    data = [day[elem] for day in days_list]

    if verbose:
        display_station_data(data)

    temp_maps = load_image_np(days_list, map_fnm_fn, width, height, color)

    X, y = gen_station_np_seq(data, timesteps)

    temp_X = gen_image_np_seq(temp_maps, width, height, timesteps)

    [
            train_X, test_X,
            temp_train_X, temp_test_X,
            train_y, test_y
    ] = split_test(
            X, temp_X, y, split=0.25)

    temp_data = [
            train_X, train_y,
            test_X, test_y,
            temp_train_X, temp_test_X]

    return temp_data


def get_min_temp_data(width=160, height=70, timesteps=10, color='hsv', verbose=False):

    fnm = 'saved_data/phys_weather/min_temp_data_%d,%d,%d,%s.p' % \
        (width, height, timesteps, color)

    if isfile(fnm):
        print('Loading min_temp_data from file')
        min_temp_data = pickle.load(open(fnm, 'rb'))

    else:
        print('Generating min_temp_data')
        min_temp_data = gen_temp_data(width, height, timesteps, verbose, color, 1)
        pickle.dump(min_temp_data, open(fnm, 'wb'))

    return min_temp_data


def get_max_temp_data(width=160, height=70, timesteps=10, color='hsv', verbose=False):

    fnm = 'saved_data/phys_weather/max_temp_data_%d,%d,%d,%s.p' % \
        (width, height, timesteps, color)

    if isfile(fnm):
        print('Loading max_temp_data from file')
        max_temp_data = pickle.load(open(fnm, 'rb'))

    else:
        print('Generating max_temp_data')
        max_temp_data = gen_temp_data(width, height, timesteps, verbose, color, 2)
        pickle.dump(max_temp_data, open(fnm, 'wb'))

    return max_temp_data
