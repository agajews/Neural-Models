from os.path import isfile

import pickle

import numpy as np

from neural_models.lib import split_test


def get_days_list(fnm, map_exists=False):
    with open(fnm) as file:
        text = file.read()

    lines = text.split('\n')[:-1]

    '''{ID: (TMIN, TMAX)}'''
    days = {}

    for line in lines[0:]:
        id = line[0:11]
        year = int(line[11:15])
        month = int(line[15:17])
        element = line[17:21]
        _char = 21
        for i in range(0, 31):
            value = int(line[_char:_char + 5]) / 10
            id = str(year) + str(month).zfill(2) + str(i + 1).zfill(2)
            if element == 'TMIN':
                try:
                    days[id] = (value, days[id][1])
                except KeyError:
                    days[id] = (value, None)
            elif element == 'TMAX':
                try:
                    days[id] = (days[id][0], value)
                except KeyError:
                    days[id] = (None, value)
            _char += 8

    days_list = []
    keys = list(days.keys())
    keys.sort()
    for day in keys:
        days_list.append((int(day), days[day][0], days[day][1]))

    def get_closest(i, days_list, position):
        j = i
        while j >= 0 and days_list[j][position] == -999.9:
            j -= 1
        j_correct = not days_list[j][position] == -999.9
        k = i
        while k < len(days_list) - 1 and days_list[k][position] == -999.9:
            k += 1
        k_correct = not days_list[k][position] == -999.9
        if j_correct and (i - j < k - i or not k_correct):
            return days_list[j][position]
        else:
            return days_list[k][position]

    for i, day in enumerate(days_list):
        if day[1] == -999.9:
            closest_min = get_closest(i, days_list, 1)
            days_list[i] = (day[0], closest_min, day[2])
        day = days_list[i]
        if day[2] == -999.9:
            closest_max = get_closest(i, days_list, 2)
            days_list[i] = (day[0], day[1], closest_max)

    f_days_list = []
    for day in days_list:
        if map_exists:
            map_fnm = 'raw_data/phys_weather/temp_maps/colormaxmin_' + \
                str(day[0]) + '.jpg'
            if isfile(map_fnm):
                f_days_list.append((day[0], round(day[1]), round(day[2])))
        else:
            f_days_list.append((day[0], round(day[1]), round(day[2])))
    return f_days_list


def gen_station_data(timesteps=10, verbose=False):
    fnm = 'saved_data/phys_weather/station_data_' + str(timesteps) + '.p'
    if isfile(fnm):
        print('Loading station_data from file')
        station_data = pickle.load(open(fnm, 'rb'))
    else:
        print('Generating station_data')
        dly_fnm = 'raw_data/phys_weather/chicago_summaries.dly'
        days_list = get_days_list(dly_fnm)
        num_days = len(days_list)
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

        min_X = np.zeros((num_days - timesteps, timesteps, min_spread))
        min_y = np.zeros((num_days - timesteps, min_spread))
        max_X = np.zeros((num_days - timesteps, timesteps, min_spread))
        max_y = np.zeros((num_days - timesteps, max_spread))
        for i in range(timesteps, num_days):
            day = days_list[i]
            example_num = i - timesteps

            min_y_pos = day[1] - min_min
            min_y[example_num, min_y_pos] = 1
            for j in range(timesteps):
                min_X_pos = days_list[example_num + j][1] - min_min
                min_X[example_num, j, min_X_pos] = 1

            max_y_pos = day[2] - max_min
            max_y[example_num, max_y_pos] = 1
            for j in range(timesteps):
                max_X_pos = days_list[example_num + j][2] - max_min
                max_X[example_num, j, max_X_pos] = 1

        [min_train_X, min_test_X,
         min_train_y, min_test_y] = split_test(min_X, min_y, split=0.25)

        [max_train_X, max_test_X,
         max_train_y, max_test_y] = split_test(max_X, max_y, split=0.25)

        station_data = [min_train_X, min_train_y, min_test_X, min_test_y,
                        max_train_X, max_train_y, max_test_X, max_test_y]

        pickle.dump(station_data, open(fnm, 'wb'))

    return station_data
