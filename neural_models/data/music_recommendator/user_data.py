import pickle

import numpy as np

from os.path import isfile

from scipy.io import wavfile

from features import mfcc

import random

import shlex

from neural_models.lib import cd

import youtube_dl


base_data_fnm = 'raw_data/music_recommendator/'
base_audio_fnm = 'raw_data/music_recommendator/audio/'
base_param_fnm = 'params/music_recommendator/'
base_save_fnm = 'saved_data/music_recommendator/'


def download(song_name, artist, song_id):

    with cd('raw_data/music_recommendator/audio'):
        ydl_opts = {
            'format': 'worstaudio',
            'outtmpl': shlex.quote(song_id + '.mp3'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download(['gvsearch1:youtube ' + song_name + ' ' + artist])
                return True
            except:
                return False


def create_wav(song_fnm, num_mels=12, song_length=2000):

    rate, wav = wavfile.read(song_fnm)
    mels = mfcc(wav, rate, numcep=num_mels, winstep=0.1)[:song_length, :]

    if wav is not None and mels.shape[0] > 10:

        wav_np = np.zeros((1, song_length, num_mels))
        wav_np[:, :mels.shape[0], :] = mels.reshape(1, mels.shape[0], mels.shape[1])

        return wav_np

    else:
        return None


def gen_song_meta(trunc_start, trunc_end):

    with open(base_data_fnm + 'unique_tracks.txt', 'r') as f:
        txt = f.read()

    song_meta = {}
    for line in txt.split('\n'):

        if len(line) > 5:

            try:
                track_id, song_id, artist, song_name = line.split('<SEP>')
                song_meta[song_id] = {'artist': artist, 'name': song_name}

            except Exception as e:
                print(e)
                print(line)

    return song_meta


def load_song_meta(mode='train', verbose=True):

    song_meta_fnm = base_save_fnm + '%s_song_meta.p' % mode
    if isfile(song_meta_fnm):

        if verbose:
            print('Loading song_meta from file')

        song_meta = pickle.load(open(song_meta_fnm, 'rb'))

    else:

        if verbose:
            print('Generating song_meta')

        if mode == 'train':
            song_meta = gen_song_meta(0, 750000)
        elif mode == 'val':
            song_meta = gen_song_meta(750000, 940000)

        pickle.dump(song_meta, open(song_meta_fnm, 'wb'))

    return song_meta


def gen_user_hist(verbose=True):

    with open(base_data_fnm + 'train_triplets.txt', 'r') as f:
        txt = f.read()

    user_hist = {}
    for line in txt.split('\n')[:5000000]:

        if len(line) > 1:

            try:
                user_id, song_id, play_count = line.split('\t')

                try:
                    user_hist[user_id]

                except:
                    user_hist[user_id] = []

                user_hist[user_id].append(
                        {'song_id': song_id, 'play_count': play_count})

            except Exception as e:
                print(e)
                print(line)

    users_ordered = list(user_hist.keys())

    return user_hist, users_ordered


def load_user_hist(verbose=True):

    user_hist_fnm = base_data_fnm + 'user_hist.p'
    users_ordered_fnm = base_data_fnm + 'users_ordered.p'

    if isfile(user_hist_fnm) and \
            isfile(users_ordered_fnm):

        if verbose:
            print('Loading user_hist and users_ordered from file')

        user_hist = pickle.load(open(user_hist_fnm, 'rb'))
        users_ordered = pickle.load(open(users_ordered_fnm, 'rb'))

    else:

        if verbose:
            print('Generating user_hist and users_ordered')

        user_hist, users_ordered = gen_user_hist(verbose)

        pickle.dump(user_hist, open(user_hist_fnm, 'wb'))
        pickle.dump(users_ordered, open(users_ordered_fnm, 'wb'))

    return user_hist, users_ordered


def gen_filtered_hist(song_meta, user_hist, users_ordered):

    filtered_hist = []
    for user in users_ordered:
        user_data = []
        for song in user_hist[user]:
            song_id = song['song_id']
            song_fnm = 'raw_data/music_recommendator/audio/%s.mp3.wav' \
                % song_id
            if song_id in song_meta and isfile(song_fnm):
                user_data.append(song)
        if len(user_data) > 5:
            filtered_hist.append(user_data)

    return filtered_hist


def load_filtered_hist(mode='train', verbose=True):

    filtered_hist_fnm = base_save_fnm + 'filtered_%s_hist.p' % mode

    if isfile(filtered_hist_fnm):

        if verbose:
            print('Loading filtered_hist from file')

        filtered_hist = pickle.load(open(filtered_hist_fnm, 'rb'))

    else:

        song_meta = load_song_meta(mode, verbose)
        user_hist, users_ordered = load_user_hist(verbose)

        if verbose:
            print('Generating filtered_hist')

        filtered_hist = gen_filtered_hist(song_meta, user_hist, users_ordered)
        pickle.dump(filtered_hist, open(filtered_hist_fnm, 'wb'))

        del song_meta, user_hist, users_ordered

    return filtered_hist


def gen_filtered_songs(song_meta):

    filtered_songs = []
    for song_id in song_meta.keys():

        song_fnm = base_audio_fnm + song_id + '.wav'

        if isfile(song_fnm):
            filtered_songs.append(song_id)

    return filtered_songs


def load_filtered_songs(mode='train', verbose=True):

    filtered_songs_fnm = base_save_fnm + 'filtered_%s_songs.p' % mode

    if isfile(filtered_songs_fnm):

        if verbose:
            print('Loading filtered_songs from file')

        filtered_songs = pickle.load(open(filtered_songs_fnm, 'rb'))

    else:

        song_meta = load_song_meta(mode, verbose)

        if verbose:
            print('Generating filtered_songs')

        filtered_songs = gen_filtered_songs(song_meta)

        pickle.dump(filtered_songs, open(filtered_songs_fnm, 'wb'))

        del song_meta

    return filtered_songs


def gen_truncated_songs(
        filtered_songs,
        trunc_start,
        trunc_end,
        shuffle=False):

    if shuffle:
        random.shuffle(filtered_songs)

    truncated_songs = filtered_songs[trunc_start:trunc_end]

    for i, song in enumerate(truncated_songs):
        truncated_songs[i] = song[:18]

    return truncated_songs


def gen_truncated_hist(
        filtered_hist,
        truncated_songs):

    songs_set = set()
    for song in truncated_songs:
        songs_set.add(song)

    truncated_hist = []
    for user in filtered_hist:

        user_data = []

        for song in user:

            song_filename = base_audio_fnm + '%s.mp3' \
                % song['song_id']

            if isfile(song_filename) and song['song_id'] in songs_set:
                user_data.append(song)

        if len(user_data) > 5:
            truncated_hist.append(user_data)

    return truncated_hist


def load_truncated_songs(
        trunc_start,
        trunc_end,
        mode='train',
        shuffle=False,
        verbose=True):

    truncated_hist_fnm = base_save_fnm + 'truncated_%s_hist_%d,%d.p' \
        % (mode, trunc_start, trunc_end)
    truncated_songs_fnm = base_save_fnm + 'truncated_%s_songs_%d,%d.p' \
        % (mode, trunc_start, trunc_end)

    if not shuffle and isfile(truncated_hist_fnm) and \
            isfile(truncated_songs_fnm):

        if verbose:
            print('Loading truncated_hist and truncated_songs from file')

        truncated_hist = pickle.load(open(truncated_hist_fnm, 'rb'))
        truncated_songs = pickle.load(open(truncated_songs_fnm, 'rb'))

    else:

        filtered_hist = load_filtered_hist(mode, verbose)
        filtered_songs = load_filtered_songs(mode, verbose)

        if verbose:
            print('Generating truncated_hist and truncated_songs')

        truncated_songs = gen_truncated_songs(
            filtered_songs,
            trunc_start,
            trunc_end,
            shuffle)
        truncated_hist = gen_truncated_hist(
            filtered_hist,
            truncated_songs)

        pickle.dump(truncated_hist, open(truncated_hist_fnm, 'wb'))
        pickle.dump(truncated_songs, open(truncated_songs_fnm, 'wb'))

        del filtered_hist, filtered_songs

    return truncated_songs, truncated_hist


def gen_wavfiles_from_song_ids(song_ids, num_mels, song_length):

    wavfiles = {}

    for i, song in enumerate(sorted(song_ids)):
        fnm = base_audio_fnm + '%s.wav' % song
        if isfile(fnm):
            if i % 10 == 0:
                print(song)
            try:
                wavfiles[song] = create_wav(fnm, num_mels, song_length)
            except:
                wavfiles[song] = None
        else:
            raise Exception('No such song %s at %s!' % (song, fnm))

    return wavfiles


def gen_data_list(truncated_hist):

    data_list = []
    for user in truncated_hist:
        for i, song in enumerate(user):
            data_entry = {}
            data_entry['user_songs_X'] = user[:i] + user[i + 1:]
            data_entry['song_X'] = song['song_id']
            data_entry['song_y'] = song['play_count']
            data_list.append(data_entry)

    return data_list


def load_all_wavfiles(mode='train', num_mels=12, song_length=2000):

    all_wavfiles_fnm = base_save_fnm + 'all_wavfiles.p'

    if isfile(all_wavfiles_fnm):
        all_wavfiles = pickle.load(open(all_wavfiles_fnm, 'rb'))

    else:
        filtered_songs = load_filtered_songs(mode='train')  # [:20]
        all_wavfiles = gen_wavfiles_from_song_ids(filtered_songs, num_mels, song_length)
        pickle.dump(all_wavfiles, open(all_wavfiles_fnm, 'wb'))

        del filtered_songs

    return all_wavfiles


def gen_wavfiles(truncated_songs, num_mels, song_length, mode='train'):

    wavfiles = {}

    all_wavfiles = load_all_wavfiles(mode, num_mels, song_length)
    # all_wavfiles = gen_wavfiles_from_song_ids(truncated_songs, num_mels)

    for song in truncated_songs:
        try:
            wavfiles[song] = all_wavfiles[song]
        except KeyError:
            wavfiles[song] = None

    del all_wavfiles

    return wavfiles


def wavfile_exists(wavfiles, song_id):

    wav_exists = True
    try:
        if wavfiles[song_id] is None:
            wav_exists = False
    except KeyError:
        wav_exists = False

    return wav_exists


def gen_wav_data_list(wavfiles, data_list):

    wav_data_list = []

    for i, entry in enumerate(data_list):

        user_songs_X = []

        for song in entry['user_songs_X']:
            song_id = song['song_id']

            if wavfile_exists(wavfiles, song_id):

                song_entry = {
                    'wav': wavfiles[song_id],
                    'play_count': song['play_count']
                }
                user_songs_X.append(song_entry)

        song_X_id = entry['song_X']

        if len(user_songs_X) > 15 and wavfile_exists(wavfiles, song_X_id):

            wav_entry = {
                'user_songs_X': user_songs_X,
                'song_X': wavfiles[song_X_id],
                'song_y': entry['song_y']
            }
            wav_data_list.append(wav_entry)

    return wav_data_list


def get_max_num_songs(wav_data_list, num_songs_cap):

    nums_of_songs = [len(example['user_songs_X']) for example in wav_data_list]
    max_num_songs = max(nums_of_songs)
    max_num_songs = min(num_songs_cap, max_num_songs)

    return max_num_songs


def get_max_song_length(wavfiles):

    max_length = 0
    for song in wavfiles.keys():
        wav = wavfiles[song]
        if wav is not None:
            max_length = max(max_length, wav.shape[1])

    return max_length


def gen_data(wav_data_list, wavfiles, num_songs_cap, num_mels, song_length):

    num_examples = len(wav_data_list)
    max_num_songs = get_max_num_songs(wav_data_list, num_songs_cap)
    # max_song_length = get_max_song_length(wavfiles)

    print(num_examples)
    print(max_num_songs)
    # print(max_song_length)

    user_songs_X = np.zeros((num_examples, max_num_songs, song_length, num_mels))
    user_count_X = np.zeros((num_examples, max_num_songs, 1))
    song_X = np.zeros((num_examples, song_length, num_mels))
    song_y = np.zeros((num_examples))

    for i, entry in enumerate(wav_data_list):
        for j, song_entry in enumerate(entry['user_songs_X'][:max_num_songs]):
            wav = song_entry['wav']
            user_songs_X[i, j, :, :] = wav
            user_count_X[i, j] = song_entry['play_count']
        wav = entry['song_X']
        song_X[i, :, :] = wav
        song_y[i] = entry['song_y']

    return user_songs_X, user_count_X, song_X, song_y


def gen_audio_dataset(
        mode='train',
        num_truncated_songs=10000,
        num_mels=6,
        song_length=2000,
        num_songs_cap=25,
        shuffle=True,
        verbose=True):

    truncated_songs, truncated_hist = load_truncated_songs(
        0,
        num_truncated_songs,
        mode=mode,
        shuffle=shuffle,
        verbose=verbose)

    data_list = gen_data_list(truncated_hist)

    if verbose:
        print('Reading songs')

    wavfiles = gen_wavfiles(truncated_songs, num_mels, song_length, mode)

    if verbose:
        print('Generating data')

    wav_data_list = gen_wav_data_list(wavfiles, data_list)

    del truncated_songs, truncated_hist

    data = gen_data(
        wav_data_list, wavfiles, num_songs_cap, num_mels, song_length)

    del wavfiles, wav_data_list

    return data
