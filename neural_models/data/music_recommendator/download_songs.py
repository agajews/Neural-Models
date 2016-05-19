from .user_data import load_data
from .user_data import download
from os import mkdir
from os.path import isfile


def download_songs():
    try:
        mkdir('audio')
    except FileExistsError:
        pass

    song_meta, user_hist, users_ordered = load_data()
    print(len(user_hist))
    failed_songs = []
    user_num = 0
    for user in users_ordered[:10000]:
        user_num += 1
        print('User number: ' + str(user_num))
        for song in user_hist[user]:
            if song is not None:
                song_id = song['song_id']
                meta = song_meta[song_id]
                print('Song meta: ' + str(meta))
                artist, song_name = meta['artist'], meta['name']
                if not isfile('audio/' + song_id + '.mp3'):
                    failed = download(artist, song_name, song_id)
                    if failed:
                        failed_songs.append(song_id)
                        print('Failed songs: ' + str(failed_songs))

if __name__ == '__main__':
    download_songs()
