import scipy.signal
from scipy.io import wavfile

from subprocess import call

from neural_models.data.music_recommendator.user_data import download

import numpy as np

from os.path import isfile


class Song(object):

    def __init__(self, song_id, name=None, artist=None, play_count=None):

        self.song_id = song_id
        self.name = name
        self.artist = artist
        self.play_count = play_count


class User(object):

    def __init__(self, user_id, songs=None):

        self.user_id = user_id
        self.hist = songs
        self.prefs = None

    def __repr__(self):

        return 'User(%d)' % self.user_id

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


def add_wavs(songs):

    for song in songs:
        add_wav(song)


def add_wav(song):

    rate, wav = wavfile.read(song.fnm)
    downsampled_size = int(wav.shape[0] * 0.01)

    if downsampled_size > 10:
        wav = scipy.signal.resample(wav, downsampled_size)

    else:
        wav = None

    if wav is not None:

        if len(wav.shape) == 2:
            bitwidth = wav.shape[1]

        else:
            bitwidth = 1

        wav_np = np.zeros((1, wav.shape[0], 3))
        wav_np[:, :, :bitwidth] = wav.reshape(1, wav.shape[0], bitwidth)

        song.wav = wav_np

    else:
        song.wav = None


def create_wavs(songs):

    for song in songs:
        song_wav_fnm = song.fnm + '.wav'
        if not isfile(song_wav_fnm):
            download(song.name, song.artist, song.song_id)
            call('lame --decode %s %s' % (song.fnm, song_wav_fnm), shell=True)
        song.fnm = song_wav_fnm


def add_filenames(songs):

    for song in songs:
        song_fnm = 'raw_data/music_recommendator/audio/%s.mp3' % song.song_id
        song.fnm = song_fnm
