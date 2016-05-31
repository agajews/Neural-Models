from subprocess import call

from neural_models.data.music_recommendator.user_data import \
    download, create_wav

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

        add_song_embeddings(model, self.hist)

    def add_filenames(self):

        add_filenames(self.hist)


def add_wavs(songs):

    for song in songs:
        add_wav(song)


def add_wav(song):

    wav = create_wav(song.fnm)
    song.wav = np.reshape(wav, (wav.shape[0], wav.shape[2], wav.shape[1]))


def create_wavs(songs):

    for song in songs:
        song_wav_fnm = song.fnm[:-4] + '.wav'
        if not isfile(song_wav_fnm):
            download(song.name, song.artist, song.song_id)
            sh = 'ffmpeg -i %s -acodec pcm_u8 -ar 22050 %s -y' % \
                (song.fnm, song_wav_fnm)
            call(sh, shell=True)
        song.fnm = song_wav_fnm


def add_filenames(songs):

    for song in songs:
        song_fnm = 'raw_data/music_recommendator/audio/%s.mp3' % song.song_id
        song.fnm = song_fnm


def add_song_embeddings(model, songs):

    print('Adding song embeddings')

    for i, song in enumerate(songs):
        if i % 100 == 0:
            print(song.song_id)
        if song.wav is not None:
            try:
                print(song.wav.shape)
                song.embedding = model.get_song_embedding(song.wav)
                del song.wav
            except Exception as e:
                if i % 1000 == 0:
                    print(e)
                # print(song.wav)
                song.embedding = None
        else:
            song.embedding = None
