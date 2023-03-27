import urllib.request
from googleapiclient.discovery import build
import pandas as pd
import sqlite3
from mhyt import yt_download
import os


api_key = 'APIKEY'
youtube = build('youtube', 'v3', developerKey=api_key)

search_queries = ['best arabesque ballet', 'ballet grand-jete slow motion',
                  'best pirouette ballet', 'ballet pa-de-bourree how to']


categories = ['arabesque', 'grand-jete', 'pirouette', 'pa-de-bourree']

video_dir = 'ballet_videos'

# download video info function


def download_video(query):
    req = youtube.search().list(q=query, part='snippet',
                                type='video', maxResults=5).execute()
    data = []
    for item in req['items']:
        item_data = dict()
        item_data['videoId'] = item['id']['videoId']
        item_data['Title'] = item['snippet']['title']
        data.append(item_data)
    df = pd.DataFrame(data)
    return df


# create sql db function
def sql_database():
    conn = sqlite3.connect('video_data.db')
    conn.execute('''DROP TABLE IF EXISTS video;''')
    conn.execute('''CREATE TABLE video
                (videoId      VARCHAR NOT NULL,
                title         VARCHAR,
                category      VARCHAR,
                PRIMARY KEY (videoId)
                );''')
    conn.commit()
    conn.close()


# insert into db function
def insert_video(videoId, title, category):
    conn = sqlite3.connect('video_data.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO video VALUES (?,?,?)",
                   (videoId, title, category))
    conn.commit()
    print('Video added successfully')
    conn.close()


# retrieve data function
def data_retrieval(video_category):
    conn = sqlite3.connect('video_data.db')
    cur = conn.cursor()
    cur.execute("SELECT videoid FROM video WHERE category=:category", {
                'category': video_category})
    result = cur.fetchall()
    return result


# create db and insert video info
sql_database()

for query in search_queries:
    df = download_video(query)
    category = query.split()[1]
    for index, p in df.iterrows():
        insert_video(p[0], p[1], category)


# download videos from db
for category in categories:
    result = data_retrieval(category)
    for i in result:
        path = os.path.join(video_dir, category)
        yt_download("https://www.youtube.com/watch?v=%s" % (i),
                    path + "/" + '%s' % (i) + '.mp4')
