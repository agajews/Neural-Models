from os.path import isfile

from cv2 import imread, resize, cvtColor, COLOR_BGR2HSV

import pickle

import numpy as np

from neural_models.lib import split_test
from .station_data import get_days_list


def gen_map_data(width=100, height=50, timesteps=10, verbose=False, color='hsv'):
    filename = 'saved_data/phys_weather/map_data_' + \
        str(width) + ',' + \
        str(height) + ',' + \
        str(timesteps) + ',' + \
        color + \
        '.p'
    if isfile(filename):
        map_data = pickle.load(open(filename, 'rb'))
    else:
        days_list = get_days_list('raw_data/phys_weather/chicago_summaries.dly', map_exists=True)
        num_days = len(days_list)
        if color == 'rgb':
            channels = 3
        elif color == 'hsv':
            channels = 1
        else:
            raise Exception('Invalid color %s' % color)

        temp_maps = np.zeros((num_days, channels, width, height))
        for i, (day, minimum, maximum) in enumerate(days_list):
            image = imread('raw_data/phys_weather/temp_maps/colormaxmin_' + str(day) + '.jpg')
            if color == 'rgb':
                image = np.transpose(image, (2, 0, 1))
                for channel in range(image.shape[0]):
                    temp_maps[i, channel, :, :] = resize(image[channel, :, :], (height, width))
            elif color == 'hsv':
                image = cvtColor(image, COLOR_BGR2HSV)
                temp_maps[i, 0, :, :] = resize(image[:, :, 0], (height, width))
        mins = [day[1] for day in days_list]
        maxs = [day[2] for day in days_list]
        min_min = min(mins)
        max_min = max(mins)
        min_max = min(maxs)
        max_max = max(maxs)

        min_spread = max_min - min_min + 1
        max_spread = max_max - min_max + 1
        if verbose:
            print('Num days: ' + str(num_days))

            print('Min min: ' + str(min_min))
            print('Max min: ' + str(max_min))
            print('Min max: ' + str(min_max))
            print('Max max: ' + str(max_max))

            print('Min spread: ' + str(min_spread))
            print('Max spread: ' + str(max_spread))

        min_map_X = np.zeros((num_days - timesteps, timesteps, min_spread))
        min_map_y = np.zeros((num_days - timesteps, min_spread))
        max_map_X = np.zeros((num_days - timesteps, timesteps, min_spread))
        max_map_y = np.zeros((num_days - timesteps, max_spread))
        temp_map_X = np.zeros((num_days - timesteps, timesteps, channels, width, height))
        for i in range(timesteps, num_days):
            day = days_list[i]
            example_num = i - timesteps

            min_map_y_pos = day[1] - min_min
            min_map_y[example_num, min_map_y_pos] = 1
            for j in range(timesteps):
                min_map_X_pos = days_list[example_num + j][1] - min_min
                min_map_X[example_num, j, min_map_X_pos] = 1

            max_map_y_pos = day[2] - max_min
            max_map_y[example_num, max_map_y_pos] = 1
            for j in range(timesteps):
                max_map_X_pos = days_list[example_num + j][2] - max_min
                max_map_X[example_num, j, max_map_X_pos] = 1

            for j in range(timesteps):
                temp_map_X[example_num, j, :, :, :] = temp_maps[example_num + j, :, :, :]

        [min_map_train_X, min_map_test_X,
         temp_map_train_X, temp_map_test_X,
         min_map_train_y, min_map_test_y] = split_test(min_map_X, temp_map_X, min_map_y, split=0.25)

        [max_map_train_X, max_map_test_X,
         temp_map_train_X, temp_map_test_X,
         max_map_train_y, max_map_test_y] = split_test(max_map_X, temp_map_X, max_map_y, split=0.25)

        map_data = [min_map_train_X, min_map_train_y, min_map_test_X, min_map_test_y,
                    max_map_train_X, max_map_train_y, max_map_test_X, max_map_test_y,
                    temp_map_train_X, temp_map_test_X]

        pickle.dump(map_data, open(filename, 'wb'))

    return map_data
