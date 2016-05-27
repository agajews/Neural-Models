from scipy.io import wavfile
from scipy import signal

import numpy as np

import pickle

from os import listdir
from os.path import isfile

from subprocess import call

from neural_models.data.music_recommendator.user_data import download
from neural_models.music_recommendator.audio_model import AudioModel


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

marisa_songs_list = [
        {
            'name': 'Baby its cold outside louis armstrong',
            'play_count': 6,
            'song_id': 'u3s5'
        },
        {
            'name': 'Summertime Ella Fitzgerald',
            'play_count': 8,
            'song_id': 'u3s9'
        },
        {
            'name': 'My name is Luca Susan Vega',
            'play_count': 10,
            'song_id': 'u3s1'
        },
        {
            'name': 'Sunshine Nora Jones',
            'play_count': 9,
            'song_id': 'u3s2'
        },
        {
            'name': 'Whats up 4 non blondes',
            'play_count': 8,
            'song_id': 'u3s3'
        },
        {
            'name': 'In this world ladysmith black mambazo',
            'play_count': 7,
            'song_id': 'u3s4'
        },
        {
            'name': 'Youve got a friend Carole King',
            'play_count': 12,
            'song_id': 'u3s6'
        },
        {
            'name': 'Malaika Angelike Kidjo',
            'play_count': 6,
            'song_id': 'u3s7'
        },
        {
            'name': 'Hijo de la luna Mecano ',
            'play_count': 12,
            'song_id': 'u3s8'
        },
        {
            'name': 'Freeway Aimee Man',
            'play_count': 9,
            'song_id': 'u3s10'
        },
        {
            'name': 'Yesterday Beatles',
            'play_count': 10,
            'song_id': 'u3s11'
        },
        {
            'name': 'Watermark Enya',
            'play_count': 6,
            'song_id': 'u3s12'
        }
]

wanqi_songs_list = [
        {
            'name': 'The Saltwater Room Owl City',
            'play_count': 6,
            'song_id': 'u4s5'
        },
        {
            'name': 'Paradise Coldplay',
            'play_count': 8,
            'song_id': 'u4s9'
        },
        {
            'name': 'Yellow Lights Of Monster and Men',
            'play_count': 5,
            'song_id': 'u4s1'
        },
        {
            'name': 'Ghost Towns Radical Face',
            'play_count': 1,
            'song_id': 'u4s2'
        },
        {
            'name': 'Welcome to Season 5 Instalok',
            'play_count': 800,
            'song_id': 'u4s3'
        },
        {
            'name': 'What a wonderful world',
            'play_count': 7,
            'song_id': 'u4s4'
        },
        {
            'name': 'Eet Regina Spector',
            'play_count': 1,
            'song_id': 'u4s6'
        },
        {
            'name': 'Vanilla Twilight Owl City',
            'play_count': 60,
            'song_id': 'u4s7'
        },
        {
            'name': 'Clocks Coldplay',
            'play_count': 12,
            'song_id': 'u4s8'
        },
        {
            'name': 'City of black and white Mat',
            'play_count': 9,
            'song_id': 'u4s10'
        },
        {
            'name': 'Westcoast Friendship Owl City',
            'play_count': 10,
            'song_id': 'u4s11'
        },
        {
            'name': 'Summer Love Workday Release',
            'play_count': 6,
            'song_id': 'u4s12'
        }
]


class Song(object):

    def __init__(self, song_id, name=None, artist=None, play_count=None):

        self.song_id = song_id
        self.name = name
        self.artist = artist
        self.play_count = play_count


class User(object):

    def __init__(self, songs_list):

        self.hist = create_songs(songs_list)

    def add_wavs(self):

        create_wavs(self.hist)
        add_wavs(self.hist)

        for song in self.hist:
            if song.wav is None:
                self.hist.remove(song)

    def add_embeddings(self, model):

        for song in self.hist:
            song.embedding = model.get_song_embedding(song.wav)

    def add_filenames(self):

        add_filenames(self.hist)


def add_wav(song):

    rate, wav = wavfile.read(song.fnm)
    downsampled_size = int(wav.shape[0] * 0.01)

    if downsampled_size > 3:
        wav = signal.resample(wav, downsampled_size)

    else:
        wav = None

    if len(wav.shape) == 2:
        bitwidth = wav.shape[1]

    else:
        bitwidth = 1

    wav_np = np.zeros((1, wav.shape[0], 3))
    wav_np[:, :, :bitwidth] = wav.reshape(1, wav.shape[0], bitwidth)

    song.wav = wav_np


def add_wavs(songs):

    for song in songs:
        add_wav(song)


'''def get_std_user_preds(model, user_songs, user_counts, song_fnm):

    input_song = get_wav(song_fnm)
    input_song_np = np.zeros((1, user_songs.shape[2], 3))
    input_song_np[0, :input_song.shape[1], :] = input_song
    preds = model.get_std_preds(input_song_np, user_songs, user_counts)

    return preds'''


def create_songs(songs_list):

    songs = []
    for song in songs_list:
        song_obj = Song(
            song['song_id'],
            song['name'],
            artist='',
            play_count=song['play_count'])

        songs.append(song_obj)

    return songs


def add_filenames(songs):

    for song in songs:
        song_fnm = 'raw_data/music_recommendator/audio/%s.mp3' % song.song_id
        song.fnm = song_fnm


def create_wavs(songs):

    for song in songs:
        song_wav_fnm = song.fnm + '.wav'
        if not isfile(song_wav_fnm):
            download(song['name'], '', song['song_id'])
            call('lame --decode %s %s' % (song.fnm, song_wav_fnm), shell=True)


'''def gen_user_data_np(songs_list):

    song_data_np = gen_song_data_np(songs_list)
    lengths = [song['wav'].shape[1] for song in song_data_np]
    user_songs = np.zeros((1, len(songs_list), max(lengths), 3))
    user_counts = np.zeros((1, len(songs_list), 1))

    for i, song in enumerate(song_data_np):
        wav = song['wav']
        user_songs[0, i, :wav.shape[1], :wav.shape[2]] = wav
        user_counts[0, i, 0] = song['play_count']

    return user_songs, user_counts'''


def add_song_embeddings(model, songs):

    for i, song in enumerate(songs):
        if i % 100 == 0:
            print(song.song_id)
        song.embedding = model.get_song_embedding(song.wav)


def gen_user_prefs(model, user):

    song_embeddings_np = np.zeros((1, len(user.hist), model.embedding))
    song_counts_np = np.zeros((1, len(user.hist), 1))

    for i, song in enumerate(user.hist):
        song_embeddings_np[:, i, :] = song.embedding
        song_counts_np[:, i, :] = song.play_count

    user_prefs = model.get_user_prefs(song_embeddings_np, song_counts_np)

    return user_prefs


def create_all_songs():

    base_fnm = 'raw_data/music_recommendator/audio/'

    all_song_fnms = listdir(base_fnm)[:20]
    all_song_fnms = [base_fnm + fnm for fnm in all_song_fnms]
    all_song_fnms = [fnm for fnm in all_song_fnms if fnm[-4:] == '.wav']
    all_song_fnms = [fnm for fnm in all_song_fnms if 'temp' not in fnm]
    all_song_fnms = [fnm for fnm in all_song_fnms if 'part' not in fnm]

    song_meta_fnm = 'saved_data/music_recommendator/song_meta.p'
    song_meta = pickle.load(open(song_meta_fnm, 'rb'))

    all_songs = []

    for i, fnm in enumerate(sorted(all_song_fnms)):
        song_id = fnm[:-8]
        song_id = song_id[-18:]

        if i % 100 == 0:
            print(song_id)

        song_name = song_meta[song_id]['name']
        artist = song_meta[song_id]['artist']

        song = Song(song_id, song_name, artist)
        song.fnm = fnm
        song.wav = add_wav(song)

        if song.wav is not None:
            all_songs.append(song)

    return all_songs


def get_all_songs_with_embeddings(model):

    print('Getting song embeddings')

    embeddings_fnm = 'saved_data/music_recommendator/all_song_embeddings.p'
    if isfile(embeddings_fnm):
        all_song_embeddings = pickle.load(open(embeddings_fnm, 'rb'))

    else:
        all_songs = create_all_songs()
        add_song_embeddings(model, all_songs)
        pickle.dump(all_song_embeddings, open(embeddings_fnm, 'wb'))

    return all_songs


'''def get_single_song_embedding(model, song_fnm):

    wav = get_wav(song_fnm)
    embedding = model.get_song_embedding(wav)

    song = {}
    song['wav'] = wav
    song['name'] = 'none'
    song['artist'] = 'none'
    song['song_id'] = 'none'
    song['embedding'] = embedding

    return [song]'''


def get_user_preds(model, user_prefs, all_songs):

    for song in all_songs:
        song.exp_play_count = model.get_preds(
            all_songs, user_prefs)


def display_preds(preds):

    for song in preds:
        print(
            'Name: %s | Artist: %s | Exp Play Count: %f' %
            (song['name'], song['artist'], song['exp_play_count']))
        # print('Embedding: %s' % song['embedding'])


def get_all_preds(model, songs_list):

    user = User(songs_list)
    user.add_filenames()
    user.add_wavs()
    user.add_embeddings()

    user_prefs = gen_user_prefs(model, user)
    # print(user_prefs)

    all_songs = get_all_songs_with_embeddings(model)

    get_user_preds(model, user_prefs, all_songs)

    def exp_count_key(k):
        return k['exp_play_count']

    all_songs = sorted(all_songs, key=exp_count_key, reverse=True)

    display_preds(all_songs[:10])


def test_pref_embedding():

    param_fnm = 'params/music_recommendator/audio_model_strict_' + \
        'n3500,l0.015,t3.p'
    model = AudioModel(param_filename=param_fnm)

    model.compile_net_notrain()
    model.build_song_embedding_fn()
    model.build_pref_embedding_fn()
    model.build_pred_fn()
    model.build_std_pred_fn()
    model.load_params()

    # get_all_preds(model, alex_songs_list)
    # get_all_preds(model, sam_songs_list)
    # get_all_preds(model, marisa_songs_list)
    get_all_preds(model, wanqi_songs_list)


def main():

    test_pref_embedding()


if __name__ == '__main__':
    main()
