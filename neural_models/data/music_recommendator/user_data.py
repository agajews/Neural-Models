import pickle

from scipy.io import wavfile
# import scipy.signal

from os import listdir
from os.path import isfile

from neural_models.lib import cd

import youtube_dl


def download(song_name, artist, song_id):

    with cd('audio'):
        ydl_opts = {
            'format': 'worstaudio',
            'outtmpl': song_id + '.mp3',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
            }],
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download(['gvsearch1:youtube ' + song_name + ' ' + artist])
                return False
            except:
                return True


def load_data(num_truncated_songs=10000):

    song_meta_fnm = 'saved_data/music_recommendator/song_meta.p'
    if isfile(song_meta_fnm):
        print('Loading song_meta from file')
        song_meta = pickle.load(open(song_meta_fnm, 'rb'))
    else:
        print('Generating song_meta')
        with open('raw_data/music_recommendator/unique_tracks.txt', 'r') as f:
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
                # download(song_name, artist, song_id)
        pickle.dump(song_meta, open(song_meta_fnm, 'wb'))

    user_hist_fnm = 'saved_data/music_recommendator/user_hist.p'
    users_ordered_fnm = 'saved_data/music_recommendator/users_ordered.p'
    if isfile(user_hist_fnm) and \
            isfile(users_ordered_fnm):
        print('Loading user_hist and users_ordered from file')
        user_hist = pickle.load(open(user_hist_fnm, 'rb'))
        users_ordered = pickle.load(open(users_ordered_fnm, 'rb'))
    else:
        print('Generating user_hist and users_ordered')
        with open('raw_data/music_recommendator/train_triplets.txt', 'r') as f:
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
        pickle.dump(user_hist, open(user_hist_fnm, 'wb'))
        pickle.dump(users_ordered, open(users_ordered_fnm, 'wb'))

    filtered_hist_fnm = 'saved_data/music_recommendator/filtered_hist.p'
    if isfile(filtered_hist_fnm):
        print('Loading filtered_hist from file')
        filtered_hist = pickle.load(open(filtered_hist_fnm, 'rb'))
    else:
        print('Generating filtered_hist')
        filtered_hist = []
        for user in users_ordered:
            user_data = []
            for song in user_hist[user]:
                if isfile('audio/' + song['song_id'] + '.mp3'):
                    user_data.append(song)
            if len(user_data) > 5:
                filtered_hist.append(user_data)
        pickle.dump(filtered_hist, open(filtered_hist_fnm, 'wb'))

    truncated_hist_fnm = 'saved_data/music_recommendator/truncated_hist_' + \
        str(num_truncated_songs) + '.p'
    truncated_songs_fnm = 'saved_data/music_recommendator/truncated_songs_' + \
        str(num_truncated_songs) + '.p'
    if isfile(truncated_hist_fnm) and \
            isfile(truncated_songs_fnm):
        print('Loading truncated_hist and truncated_songs from file')
        truncated_hist = pickle.load(open(truncated_hist_fnm, 'rb'))
        truncated_songs = pickle.load(open(truncated_songs_fnm, 'rb'))
    else:
        print('Generating truncated_hist and truncated_songs')
        truncated_songs = listdir(
                'raw_data/music_recommendator/audio')[:num_truncated_songs]
        for i, song in enumerate(truncated_songs):
            truncated_songs[i] = song[:18]
        songs_set = set()
        for song in truncated_songs:
            songs_set.add(song)

        truncated_hist = []
        for user in filtered_hist:
            user_data = []
            song_filename = 'raw_data/music_recommendator/audio/' + \
                song['song_id'] + '.mp3'
            for song in user:
                if isfile(song_filename) and song['song_id'] in songs_set:
                    user_data.append(song)
            if len(user_data) > 5:
                truncated_hist.append(user_data)
        pickle.dump(truncated_hist, open(truncated_hist_fnm, 'wb'))
        pickle.dump(truncated_songs, open(truncated_songs_fnm, 'wb'))

    return [song_meta, user_hist, users_ordered,
            filtered_hist, truncated_songs, truncated_hist]


def gen_audio_dataset(num_truncated_songs=10000, num_mels=24):

    _, _, _, _, truncated_songs, truncated_hist = load_data(
            num_truncated_songs=num_truncated_songs)

    data_list = []
    for user in truncated_hist:
        for i, song in enumerate(user):
            data_entry = {}
            data_entry['user_songs_X'] = user[:i] + user[i + 1:]
            data_entry['song_X'] = song['song_id']
            data_entry['song_y'] = song['play_count']
            data_list.append(data_entry)

    wavfiles = {}
    print('Reading songs')
    for song in truncated_songs:
        fnm = 'raw_data/music_recommendator/audio/' + song + '.mp3.wav'
        if isfile(fnm):
            rate, wav = wavfile.read(fnm)
            print(wav)
            wavfiles[song] = {
                    'wav': wav,
                    'rate': rate}
        else:
            raise Exception('No such song %s at %s!' % (song, fnm))

    wav_data_list = []
    print('Generating data')
    for i, entry in enumerate(data_list):
        data_entry = {}
        user_songs_X = []
        for song in entry['user_songs_X']:
            song_entry = {}
            song_entry['wav'] = wavfiles[song['song_id']]['wav']
            song_entry['play_count'] = song['play_count']
            user_songs_X.append(song_entry)
        wav_entry = {}
        wav_entry['user_songs_X'] = user_songs_X
        wav_entry['song_X'] = wavfiles[entry['song_X']]
        wav_entry['song_y'] = entry['song_y']
        wav_data_list.append(wav_entry)
    return wav_data_list
