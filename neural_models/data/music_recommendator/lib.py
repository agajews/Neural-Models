from neural_models.music_recommendator.test_audio_model import \
    add_wavs, create_wavs, add_filenames


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
