from scipy.io import wavfile
from scipy import signal

import numpy as np

import pickle

from os import listdir
from os.path import isfile

from subprocess import call

from neural_models.data.music_recommendator.user_data import download
from neural_models.music_recommendator.audio_model import AudioModel


def get_wav(song_fnm):

    rate, wav = wavfile.read(song_fnm)
    downsampled_size = int(wav.shape[0] * 0.01)
    if downsampled_size > 3:
        wav = signal.resample(wav, downsampled_size)

    else:
        return None

    if len(wav.shape) == 2:
        bitwidth = wav.shape[1]

    else:
        bitwidth = 1

    wav_np = np.zeros((1, wav.shape[0], 3))
    wav_np[:, :, :bitwidth] = wav.reshape(1, wav.shape[0], bitwidth)

    return wav_np


def get_std_user_preds(model, user_songs, user_counts, song_fnm):

    input_song = get_wav(song_fnm)
    input_song_np = np.zeros((1, user_songs.shape[2], 3))
    input_song_np[0, :input_song.shape[1], :] = input_song
    preds = model.get_std_preds(input_song_np, user_songs, user_counts)

    return preds


def gen_song_data_np(songs_list):

    song_data_np = []

    for song in songs_list:

        song_fnm = 'raw_data/music_recommendator/audio/%s.mp3' % song['song_id']
        song_wav_fnm = song_fnm + '.wav'

        if not isfile(song_wav_fnm):
            download(song['name'], '', song['song_id'])
            call('lame --decode %s %s' % (song_fnm, song_wav_fnm), shell=True)

        wav = get_wav(song_wav_fnm)

        song_wav = {}
        song_wav['wav'] = wav
        song_wav['play_count'] = song['play_count']
        song_wav['name'] = song['name']
        song_wav['song_id'] = song['song_id']

        if wav is not None:
            song_data_np.append(song_wav)

    return song_data_np


def gen_user_data_np(songs_list):

    song_data_np = gen_song_data_np(songs_list)
    lengths = [song['wav'].shape[1] for song in song_data_np]
    user_songs = np.zeros((1, len(songs_list), max(lengths), 3))
    user_counts = np.zeros((1, len(songs_list), 1))

    for i, song in enumerate(song_data_np):
        wav = song['wav']
        user_songs[0, i, :wav.shape[1], :wav.shape[2]] = wav
        user_counts[0, i, 0] = song['play_count']

    return user_songs, user_counts


def gen_song_embeddings(model, song_data_np):

    for i, song_data in enumerate(song_data_np):
        if i % 100 == 0:
            print(song_data['song_id'])
        song_data['embedding'] = model.get_song_embedding(song_data['wav'])

    return song_data_np


def gen_user_prefs(model, song_embeddings):

    song_embeddings_np = np.zeros((1, len(song_embeddings), model.embedding))
    song_counts_np = np.zeros((1, len(song_embeddings), 1))

    for i, song_data in enumerate(song_embeddings):
        song_embeddings_np[:, i, :] = song_data['embedding']
        song_counts_np[:, i, :] = song_data['play_count']

    user_prefs = model.get_user_prefs(song_embeddings_np, song_counts_np)

    return user_prefs


def get_all_song_wavs():

    print('getting wavs')

    base_fnm = 'raw_data/music_recommendator/audio/'
    all_song_fnms = listdir(base_fnm)
    all_song_fnms = [base_fnm + fnm for fnm in all_song_fnms]
    all_song_fnms = [fnm for fnm in all_song_fnms if fnm[-4:] == '.wav']
    all_song_fnms = [fnm for fnm in all_song_fnms if 'temp' not in fnm]
    all_song_fnms = [fnm for fnm in all_song_fnms if 'part' not in fnm]

    song_meta_fnm = 'saved_data/music_recommendator/song_meta.p'
    song_meta = pickle.load(open(song_meta_fnm, 'rb'))

    all_song_wavs = []

    for i, fnm in enumerate(sorted(all_song_fnms)):
        song_id = fnm[:-8]
        song_id = song_id[-18:]

        if i % 100 == 0:
            print(song_id)

        try:
            song_wav = {}
            song_wav['wav'] = get_wav(fnm)
            song_wav['name'] = song_meta[song_id]['name']
            song_wav['artist'] = song_meta[song_id]['artist']
            song_wav['song_id'] = song_id
        except:
            song_wav['wav'] = None

        if song_wav['wav'] is not None:
            all_song_wavs.append(song_wav)

    return all_song_wavs


def get_all_song_embeddings(model):

    print('Getting song embeddings')

    all_song_embeddings_fnm = 'saved_data/music_recommendator/all_song_embeddings.p'
    if isfile(all_song_embeddings_fnm):
        all_song_embeddings = pickle.load(open(all_song_embeddings_fnm, 'rb'))

    else:
        all_song_wavs = get_all_song_wavs()
        all_song_embeddings = gen_song_embeddings(model, all_song_wavs)
        pickle.dump(all_song_embeddings, open(all_song_embeddings_fnm, 'wb'))

    return all_song_embeddings


def get_user_preds(model, user_prefs, all_song_embeddings):

    songs = []
    for song_embedding in all_song_embeddings:
        exp_play_count = model.get_preds(song_embedding['embedding'], user_prefs)[0]

        song = {}
        song['name'] = song_embedding['name']
        song['artist'] = song_embedding['artist']
        song['song_id'] = song_embedding['song_id']
        song['exp_play_count'] = exp_play_count

        songs.append(song)

    return songs


def test_pref_embedding():

    alex_songs_list = [
            {
                'name': 'hamilton room where it happens',
                'play_count': 15,
                'song_id': 'u1s1'
            },
            {
                'name': 'hamilton hurricane',
                'play_count': 10,
                'song_id': 'u1s2'
            },
            {
                'name': 'pentatonix mary did you know',
                'play_count': 10,
                'song_id': 'u1s3'
            },
            {
                'name': 'pentatonix daft funk',
                'play_count': 10,
                'song_id': 'u1s4'
            },
            {
                'name': 'ellie goulding lights',
                'play_count': 7,
                'song_id': 'u1s5'
            },
            {
                'name': 'clean bandit rather be',
                'play_count': 7,
                'song_id': 'u1s6'
            },
            {
                'name': 'beatles penny lane',
                'play_count': 5,
                'song_id': 'u1s7'
            },
            {
                'name': 'beatles let it be',
                'play_count': 5,
                'song_id': 'u1s8'
            },
            {
                'name': 'queen bohemian rhapsody',
                'play_count': 5,
                'song_id': 'u1s9'
            },
            {
                'name': 'queen under pressure',
                'play_count': 5,
                'song_id': 'u1s10'
            },
            {
                'name': 'fall out boy uma thurman',
                'play_count': 4,
                'song_id': 'u1s11'
            },
            {
                'name': 'jackson black or white',
                'play_count': 4,
                'song_id': 'u1s12'
            }
    ]

    sam_songs_list = [
            {
                'name': 'Pt.2 Kanye West ',
                'play_count': 10,
                'song_id': 'u2s1'
            },
            {
                'name': 'Fire and Rain James Taylor',
                'play_count': 10,
                'song_id': 'u2s2'
            },
            {
                'name': 'Sun King Beatles',
                'play_count': 12,
                'song_id': 'u2s3'
            },
            {
                'name': 'Fly me to the Moon Sinatra ',
                'play_count': 3,
                'song_id': 'u2s4'
            },
            {
                'name': 'Queen Bohemian Rhapsody',
                'play_count': 3,
                'song_id': 'u2s5'
            },
            {
                'name': 'Ultralight Beam Kanye West',
                'play_count': 12,
                'song_id': 'u2s6'
            },
            { 
                'name': 'Coco Butter Kisses Chance the rapper',
                'play_count': 15,
                'song_id': 'u2s7'
            },
            {
                'name': 'Sunday Candy Social Experiment ',
                'play_count': 15,
                'song_id': 'u2s8'
            },
            {
                'name': 'Earned It The weeknd',
                'play_count': 8,
                'song_id': 'u2s9'
            },
            {
                'name': 'Thinkin Bout you frank ocean ',
                'play_count': 9,
                'song_id': 'u2s10'
            },
            {
                'name': ' She came in through the bathroom window beatles',
                'play_count': 10,
                'song_id': 'u2s11'
            },
            {
                'name': 'Carolina in my mind james taylor ',
                'play_count': 6,
                'song_id': 'u2s12'
            }
    ]

    model = AudioModel(param_filename='params/music_recommendator/audio_model_strict_n3500,l0.015,t2.p')
    model.compile_net_notrain()
    model.build_song_embedding_fn()
    model.build_pref_embedding_fn()
    model.build_pred_fn()
    model.build_std_pred_fn()
    model.load_params()

    song_data_np = gen_song_data_np(sam_songs_list)

    song_embeddings = gen_song_embeddings(model, song_data_np)

    user_prefs = gen_user_prefs(model, song_embeddings)
    print(user_prefs)

    all_song_embeddings = get_all_song_embeddings(model)

    user_preds = get_user_preds(model, user_prefs, all_song_embeddings)
    user_preds = sorted(user_preds, key=lambda k: k['exp_play_count'], reverse=False)

    print(user_preds[:10])

    '''user_songs, user_counts = gen_user_data_np(songs_list)
    input_song_fnm = 'raw_data/music_recommendator/audio/' + 'SOAATLI12A8C13E319.mp3.wav'
    user_preds = get_std_user_preds(model, user_songs, user_counts, input_song_fnm)
    print(user_preds)'''


def main():

    test_pref_embedding()


if __name__ == '__main__':
    main()
