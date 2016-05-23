from lasagne import init
from lasagne.layers import InputLayer, LSTMLayer, \
    DropoutLayer, SliceLayer, DenseLayer, ConcatLayer
from lasagne.layers import get_output, get_all_layers
from lasagne.nonlinearities import tanh, softmax

import theano

import shlex

from scipy.io import wavfile
from scipy import signal

import numpy as np

import pickle

from os import listdir
from os.path import isfile

from subprocess import call

from neural_models.data.music_recommendator.user_data import \
    gen_audio_dataset, download

from neural_models.lib import split_val, net_on_seq

from neural_models.models import RegressionModel


class AudioModel(RegressionModel):

    def get_default_param_filename(self):

        return 'params/music_recommendator/audio_model.p'

    def load_hyperparams(self, hyperparams):

        self.load_data_hyperparams(hyperparams)
        self.load_train_hyperparams(hyperparams)
        self.load_net_hyperparams(hyperparams)

    def load_net_hyperparams(self, hyperparams):

        self.num_hidden = int(hyperparams['num_hidden'])
        self.dropout_val = float(hyperparams['dropout_val'])
        self.grad_clip = float(hyperparams['grad_clip'])
        self.l2_reg_weight = float(hyperparams['l2_reg_weight'])

    def load_data_hyperparams(self, hyperparams):

        self.num_songs = int(hyperparams['num_songs'])
        self.bitwidth = int(hyperparams['bitwidth'])

    def load_train_hyperparams(self, hyperparams):

        self.num_epochs = int(hyperparams['num_epochs'])
        self.learning_rate = float(hyperparams['learning_rate'])
        self.batch_size = int(hyperparams['batch_size'])

    def load_default_hyperparams(self):

        self.load_default_data_hyperparams()
        self.load_default_train_hyperparams()
        self.load_default_net_hyperparams()

    def load_default_data_hyperparams(self):

        self.bitwidth = 3
        self.num_songs = 3500

    def load_default_net_hyperparams(self):

        self.num_hidden = 100
        self.dropout_val = 0.2
        self.grad_clip = 927
        self.l2_reg_weight = 0.0007
        self.embedding = 100

    def load_default_train_hyperparams(self):

        self.num_epochs = 5
        self.batch_size = 12
        self.learning_rate = 0.015

    def create_lstm_stack(self, net):

        net = LSTMLayer(
                net, self.num_hidden,
                grad_clipping=self.grad_clip,
                nonlinearity=tanh)

        net = DropoutLayer(net, self.dropout_val)

        return net

    def create_song_embedding(self):

        # shape=(num_songs, song_length, bitwidth)
        i_song = InputLayer(shape=(None, None, self.bitwidth))
        self.i_song = i_song
        net = i_song

        for _ in range(1):
            net = self.create_lstm_stack(net)

        # output_shape=(num_songs, embedding)
        net = SliceLayer(net, -1, 1)
        net = DenseLayer(
                net,
                num_units=self.embedding,
                W=init.Normal(),
                nonlinearity=tanh)

        return net, i_song

    def create_song_encoder(self, l_song_embedding):

        # shape=(num_users, num_songs, song_length, bitwidth)
        i_user_songs = InputLayer(shape=(
            None, None, None, self.bitwidth))

        l_song_encoder = net_on_seq(l_song_embedding, i_user_songs)
        # output_shape=(num_users, num_songs, embedding)

        return l_song_encoder, i_user_songs

    def create_pref_embedding(self, l_song_vals):

        # shape=(num_users, num_songs, embedding + 1 (value is play_count))
        net = l_song_vals

        for _ in range(1):
            net = self.create_lstm_stack(net)

        # output_shape=(num_users, embedding)
        net = SliceLayer(net, -1, 1)
        net = DenseLayer(
                net,
                num_units=self.embedding,
                W=init.Normal(),
                nonlinearity=tanh)

        self.pref_embedding = net

        return net

    def create_user_pref_encoder(self, l_song_embedding):

        # shape=(num_users, num_songs, embedding)
        l_song_encoder, i_user_songs = self.create_song_encoder(l_song_embedding)

        # shape=(num_users, num_songs, 1 (value is play_count))
        i_user_counts = InputLayer(shape=(
            None, None, 1))

        # shape=(num_users, num_songs, embedding + 1 (value is play_count))
        l_song_vals = ConcatLayer([i_user_counts, l_song_encoder], axis=2)

        # output_shape=(num_users, embedding)
        l_user_prefs = self.create_pref_embedding(l_song_vals)

        return l_user_prefs, i_user_songs, i_user_counts

    def create_model(self):

        # shape=(num_users, embedding)
        l_song_embedding, i_input_song = self.create_song_embedding()
        self.layers += get_all_layers(l_song_embedding)
        self.i_input_song = i_input_song
        self.song_embedding = l_song_embedding

        i_input_song_embedding = InputLayer((None, self.embedding), input_var=get_output(l_song_embedding))
        self.i_input_song_embedding = i_input_song_embedding

        # shape=(num_users, embedding)
        l_user_prefs, i_user_songs, i_user_counts = \
            self.create_user_pref_encoder(l_song_embedding)
        self.i_user_songs = i_user_songs
        self.i_user_counts = i_user_counts
        self.layers += get_all_layers(l_user_prefs)

        i_prefs = InputLayer((None, self.embedding), input_var=get_output(l_user_prefs))
        self.i_prefs = i_prefs

        # shape=(num_users, 2*embedding)
        net = ConcatLayer([i_prefs, i_input_song_embedding], axis=1)

        net = DenseLayer(
                net,
                num_units=self.num_hidden,
                W=init.Normal(),
                nonlinearity=tanh)

        net = DenseLayer(
                net,
                num_units=1,
                W=init.Normal(),
                nonlinearity=softmax)
        net = SliceLayer(net, 0, 1)

        self.net = net
        self.layers += get_all_layers(net)

        return [i_user_songs, i_user_counts, i_input_song]

    def build_song_embedding_fn(self):

        embedding_out = get_output(self.song_embedding)
        embedding_fn = theano.function(
            [self.i_input_song.input_var], embedding_out,
            allow_input_downcast=True)

        self.get_song_embedding = embedding_fn

    def build_pref_embedding_fn(self):

        pref_out = get_output(self.pref_embedding)
        pref_fn = theano.function(
            [self.i_user_songs.input_var, self.i_user_counts.input_var],
            pref_out,
            allow_input_downcast=True)

        self.get_pref_embedding = pref_fn

    def build_pred_fn(self):

        pred_out = get_output(self.net)
        pred_fn = theano.function(
            [self.i_input_song_embedding.input_var, self.i_prefs.input_var],
            pred_out,
            allow_input_downcast=True)

        self.get_preds = pred_fn

    def get_supp_model_params(self, train_Xs, train_y, val_Xs, val_y):

        return None

    def train_with_data(self):

        data = self.get_data()

        self.train_model(
                *data,
                save=True, epoch_save=True,
                verbose=True, val=True)

    def get_data(self):

        [
                train_user_songs_X, train_user_count_X,
                train_song_X, train_y,
                unused_test, unused_test, unused_test, unused_test
        ] = gen_audio_dataset(num_truncated_songs=self.num_songs)

        [
                train_user_songs_X, val_user_songs_X,
                train_user_count_X, val_user_count_X,
                train_song_X, val_song_X,
                train_y, val_y
        ] = split_val(
                train_user_songs_X, train_user_count_X,
                train_song_X, train_y,
                split=0.25)

        train_Xs = [
                train_user_songs_X,
                train_user_count_X,
                train_song_X]

        val_Xs = [
                val_user_songs_X,
                val_user_count_X,
                val_song_X]

        return train_Xs, val_Xs, train_y, val_y


def train_default():

    model = AudioModel(param_filename='params/music_recommendator/audio_model_strict_n3500,l0.015.p')
    model.train_with_data()


def get_wav(song_fnm):

    rate, wav = wavfile.read(song_fnm)
    downsampled_size = int(wav.shape[0] * 0.10)
    wav = signal.resample(wav, downsampled_size)

    if len(wav.shape) == 2:
        bitwidth = wav.shape[1]

    else:
        bitwidth = 1

    wav_np = np.zeros((1, wav.shape[0], 3))
    wav_np[:, :, :bitwidth] = wav.reshape(1, wav.shape[0], bitwidth)

    return wav_np


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

        song_data_np.append(song_wav)

    return song_data_np


def gen_song_embeddings(model, song_data_np):

    for song_data in song_data_np:
        song_data['embedding'] = model.get_song_embedding(song_data['wav'])

    return song_data_np


def gen_user_prefs(model, song_embeddings, song_data_np):

    song_embeddings_np = np.zeros((1, len(song_embeddings), model.embedding))
    song_counts_np = np.zeros((1, len(song_embeddings), 1))

    for i, [song_embedding, song_data] in enumerate(zip(song_embeddings, song_data_np)):
        song_embeddings_np[:, i, :] = song_embedding['embedding']
        song_counts_np = song_data['play_count']

    user_prefs = model.get_user_prefs(song_embeddings_np, song_counts_np)

    return user_prefs


def get_all_song_wavs():

    all_song_fnms = listdir('raw_data/music_recommendator/audio')
    all_song_fnms = [fnm for fnm in all_song_fnms if fnm[-3:] == '.wav']

    song_meta_fnm = 'saved_data/music_recommendator/song_meta.p'
    song_meta = pickle.load(open(song_meta_fnm, 'rb'))

    all_song_wavs = []

    for fnm in all_song_fnms:
        song_wav = {}
        song_wav['wav'] = get_wav(fnm)
        song_wav['name'] = song_meta[fnm[:-8]]

        all_song_wavs.append(song_wav)

    return all_song_wavs


def get_user_preds(model, user_prefs, all_song_embeddings):

    for song_embedding in all_song_embeddings:
        exp_play_count = model.get_preds(song_embedding['embedding'], user_prefs)

        song_embedding['exp_play_count'] = exp_play_count

    return all_song_embeddings


def test_pref_embedding():

    songs_list = [
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

    model = AudioModel(param_filename='params/music_recommendator/audio_model_strict_n3500,l0.015.p')
    model.compile_net_notrain()
    model.build_song_embedding_fn()
    model.build_pref_embedding_fn()
    model.build_pred_fn()
    model.load_params()

    song_data_np = gen_song_data_np(songs_list)

    song_embeddings = gen_song_embeddings(model, song_data_np)

    user_prefs = gen_user_prefs(model, song_embeddings)

    all_song_wavs = get_all_song_wavs()
    all_song_embeddings = gen_song_embeddings(model, all_song_wavs)

    user_preds = get_user_preds(model, user_prefs, all_song_embeddings)
    user_preds = sorted(user_preds, key=lambda k: k['exp_play_count'], reverse=True)

    print(user_preds[:10])


def main():

    test_pref_embedding()

    # train_default()

    # bayes_hyper_optim_station()

    # grid_hyper_optim_station()


if __name__ == '__main__':
    main()
