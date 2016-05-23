from os.path import isfile

import pickle

import numpy as np

from neural_models.lib import split_test


def extract_line_meta(line):

    station = line[0:11]
    year = line[11:15]
    month = line[15:17].zfill(2)
    element = line[17:21].zfill(2)

    return station, year, month, element


def read_dly(text, elements):

    lines = text.split('\n')[:-1]
    days = {}

    for line in lines[0:]:
        stat_id, year, month, elem = extract_line_meta(line)

        for day_num, day_char in enumerate(range(21, 21 + 8 * 31, 8)):
            value = int(line[day_char:day_char + 5])
            value /= 10

            day_id = year + month + str(day_num + 1).zfill(2)
            if elem in elements:
                try:
                    days[day_id][elem] = value
                except KeyError:
                    days[day_id] = {elem: value}

    return days


def build_days_list(days):

    days_list = []
    keys = list(days.keys())
    keys.sort()

    for day in keys:
        item = (int(day), days[day]['TMIN'], days[day]['TMAX'])
        days_list.append(item)

    return days_list


def find_closest_filled_index(i, item_list, elem, empty_val):

    for distance in range(len(item_list)):
        left_check = i - distance
        if left_check >= 0:
            if not item_list[left_check][elem] == empty_val:
                return left_check

        right_check = i + distance
        if right_check < len(item_list):
            if not item_list[right_check][elem] == empty_val:
                return right_check

    return None


def fill_empty_values(item_list, elements, empty_val):

    for i, item in enumerate(item_list):
        for elem in elements:
            if item[elem] == empty_val:
                closest_index = find_closest_filled_index(
                    i, item_list, elem, empty_val)
                item_list[i] = item_list[closest_index]

    return item_list


def round_list_of_tuples(item_list, elements):

    for item_num, item in enumerate(item_list):
        new_item = []
        for elem_num, item_elem in enumerate(item):
            if elem_num in elements:
                new_item.append(round(item_elem))
            else:
                new_item.append(item_elem)

        item_list[item_num] = tuple(new_item)

    return item_list


def filter_list_with_file(item_list, fnm_fn):

    new_item_list = []

    for i, item in enumerate(item_list):
        if isfile(fnm_fn(item)):
            new_item_list.append(item)

    return new_item_list


def map_fnm_fn(day):

    return 'raw_data/phys_weather/temp_maps/colormaxmin_%d.jpg' % day[0]


def get_days_list(fnm, map_filter=False):

    with open(fnm) as file:
        text = file.read()

    days = read_dly(text, ['TMIN', 'TMAX'])
    days_list = build_days_list(days)

    days_list = fill_empty_values(days_list, [1, 2], -999.9)
    days_list = round_list_of_tuples(days_list, [1, 2])

    if map_filter:
        days_list = filter_list_with_file(days_list, map_fnm_fn)

    return days_list


def get_data_meta(data_list):

    num = len(data_list)
    small = min(data_list)
    large = max(data_list)
    spread = large - small + 1

    return num, small, large, spread


def gen_station_np_seq(data_list, timesteps):

    num_days, smallest, largest, spread = \
        get_data_meta(data_list)

    X = np.zeros((num_days - timesteps, timesteps, spread))
    y = np.zeros((num_days - timesteps, spread))

    for day_num in range(timesteps, num_days):
        min_val = data_list[day_num]
        example_num = day_num - timesteps

        y_category = min_val - smallest
        y[example_num, y_category] = 1

        for hist_num in range(timesteps):
            hist_val = data_list[example_num + hist_num]
            hist_category = hist_val - smallest
            X[example_num, hist_num, hist_category] = 1

    return X, y


def display_station_data(data):

    num_days, smallest, largest, spread = \
        get_data_meta(data)

    print('Num days: %d' % num_days)

    print('Min: %d' % smallest)
    print('Max: %d' % largest)

    print('Spread: %d' % spread)


def gen_station_data(timesteps, verbose, elem):

    dly_fnm = 'raw_data/phys_weather/chicago_summaries.dly'
    days_list = get_days_list(dly_fnm)

    data = [day[elem] for day in days_list]

    if verbose:
        display_station_data(data)

    X, y = gen_station_np_seq(data, timesteps)

    [
            train_X, test_X,
            train_y, test_y
    ] = split_test(X, y, split=0.25)

    station_data = [
            train_X, train_y, test_X, test_y]

    return station_data


def get_min_station_data(timesteps=10, verbose=False):

    fnm = 'saved_data/phys_weather/min_station_data_%d.p' % timesteps

    if isfile(fnm):
        print('Loading min_station_data from file')
        min_station_data = pickle.load(open(fnm, 'rb'))

    else:
        print('Generating station_data')
        min_station_data = gen_station_data(timesteps, verbose, 1)
        pickle.dump(min_station_data, open(fnm, 'wb'))

    return min_station_data


def get_max_station_data(timesteps=10, verbose=False):

    fnm = 'saved_data/phys_weather/max_station_data_%d.p' % timesteps

    if isfile(fnm):
        print('Loading max_station_data from file')
        max_station_data = pickle.load(open(fnm, 'rb'))

    else:
        print('Generating station_data')
        max_station_data = gen_station_data(timesteps, verbose, 2)
        pickle.dump(max_station_data, open(fnm, 'wb'))

    return max_station_data
