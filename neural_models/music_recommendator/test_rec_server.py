from requests import get, post

from time import sleep


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


user_id = 'u5'

data = [('user_id', user_id)]

adduser_url = 'http://52.38.71.139:8000/adduser'
users_url = 'http://52.38.71.139:8000/users'
recs_url = 'http://52.38.71.139:8000/recs/%s' % user_id

for song in alex_songs_list:

    song_name = ('song_name', song['name'])
    song_artist = ('song_artist', '')
    song_id = ('song_id', song['song_id'])
    play_count = ('play_count', song['play_count'])

    data.append(song_name)
    data.append(song_artist)
    data.append(song_id)
    data.append(play_count)

res = post(adduser_url, data)
print(res.json())

res = get(users_url)
print(res.json())

success = False
while not success:
    res = get(recs_url).json()

    if res['message'] == 'success':
        success = True

    print(res)
    sleep(1)
