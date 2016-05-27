
import numpy as np

import pickle

from os import listdir  # , kill, getpid
from os.path import isfile

# import signal

import neural_models
from neural_models.music_recommendator.audio_model import AudioModel
from neural_models.data.music_recommendator.lib import add_wav


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


def create_songs(songs_list):

    songs = []
    for song in songs_list:
        song_obj = neural_models.data.music_recommendator.lib.Song(
            song['song_id'],
            song['name'],
            artist='',
            play_count=song['play_count'])

        songs.append(song_obj)

    return songs


def add_song_embeddings(model, songs):

    for i, song in enumerate(songs):
        if i % 100 == 0:
            print(song.song_id)
        if song.wav is not None:
            try:
                song.embedding = model.get_song_embedding(song.wav)
            except:
                print(song.wav.shape)
                song.embedding = None
        else:
            song.embedding = None


def gen_user_prefs(model, user):

    song_embeddings_np = np.zeros((1, len(user.hist), model.embedding))
    song_counts_np = np.zeros((1, len(user.hist), 1))

    for i, song in enumerate(user.hist):
        song_embeddings_np[:, i, :] = song.embedding
        song_counts_np[:, i, :] = song.play_count

    user.prefs = model.get_user_prefs(song_embeddings_np, song_counts_np)


def create_all_songs():

    base_fnm = 'raw_data/music_recommendator/audio/'

    all_song_fnms = listdir(base_fnm)  # [:20]
    all_song_fnms = [base_fnm + fnm for fnm in all_song_fnms]
    all_song_fnms = [fnm for fnm in all_song_fnms if fnm[-4:] == '.wav']

    song_meta_fnm = 'saved_data/music_recommendator/song_meta.p'
    song_meta = pickle.load(open(song_meta_fnm, 'rb'))

    all_songs = []

    for i, fnm in enumerate(sorted(all_song_fnms)):
        song_id = fnm[:-8]
        song_id = song_id[-18:]

        try:
            song_name = song_meta[song_id]['name']
            artist = song_meta[song_id]['artist']

            song = neural_models.data.music_recommendator.lib.Song(song_id, song_name, artist)
            song.fnm = fnm
            add_wav(song)

        except:
            song.wav = None

        if song.wav is not None:

            if i % 100 == 0:
                print(song_id)

            all_songs.append(song)

    return all_songs


def filter_songs_by_embedding(all_songs):

    for song in all_songs:
        if song.embedding is None:
            all_songs.remove(song)


def get_all_songs_with_embeddings(model):

    print('Getting song embeddings')

    embeddings_fnm = 'saved_data/music_recommendator/all_song_embeddings.p'
    if isfile(embeddings_fnm):
        print('Loading pickled file')
        all_songs = pickle.load(open(embeddings_fnm, 'rb'))
        print('Loaded saved file')

    else:
        all_songs = create_all_songs()
        add_song_embeddings(model, all_songs)
        filter_songs_by_embedding(all_songs)
        pickle.dump(all_songs, open(embeddings_fnm, 'wb'))

    return all_songs


def get_user_preds(model, user, all_songs):

    for song in all_songs:
        song.exp_play_count = model.get_preds(
            song.embedding, user.prefs)


def display_preds(preds):

    print('Displaying preds')

    for song in preds:
        print(
            'Name: %s | Artist: %s | Exp Play Count: %f' %
            (song.name, song.artist, song.exp_play_count))
        # print('Embedding: %s' % song.embedding)


def get_user_recs(user, model):

    if user.prefs is None:
        print('Generating prefs')
        gen_user_prefs(model, user)

    print('Getting embeddings')
    all_songs = get_all_songs_with_embeddings(model)

    print('Getting preds')
    get_user_preds(model, user, all_songs)

    def exp_count_key(s):
        return s.exp_play_count

    all_songs = sorted(all_songs, key=exp_count_key, reverse=True)

    return all_songs[:10]


def get_all_preds(model, user_id, songs_list):

    user = neural_models.data.music_recommendator.lib.User(user_id)
    user.hist = create_songs(songs_list)
    user.add_filenames()
    user.add_wavs()
    user.add_embeddings(model)

    recs = get_user_recs(user, model)

    display_preds(recs)


def setup_test_model():

    param_fnm = 'params/music_recommendator/audio_model_strict_' + \
        'n3500,l0.015,t3.p'
    model = AudioModel(param_filename=param_fnm)

    model.compile_net_notrain()
    model.build_song_embedding_fn()
    model.build_pref_embedding_fn()
    model.build_pred_fn()
    model.build_std_pred_fn()
    model.load_params()

    return model


def test_pref_embedding():

    model = setup_test_model()

    # get_all_preds(model, alex_songs_list)
    # get_all_preds(model, sam_songs_list)
    # get_all_preds(model, marisa_songs_list)
    get_all_preds(model, 'u4', wanqi_songs_list)


def main():

    test_pref_embedding()


if __name__ == '__main__':
    main()
