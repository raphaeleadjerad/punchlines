# Download lyrics


# modules
import glob
import time
import lyricsgenius
import json
import pandas as pd
from nltk import RegexpTokenizer
from collections import defaultdict
import os

# Parameters
url_file = "data/songlist.txt"
tokenizer = RegexpTokenizer(r"\w+")

# scrape Genius website with API token
token = pd.read_csv("data/token.txt")
genius = lyricsgenius.Genius(str(token["token"][0]))


# import list of url


def clean_list_url(file):
    urls = pd.read_csv(file)
    urls.columns = ["url"]
    urls["url"] = urls["url"].str.strip()

    urls["artist"] = (urls["url"].str.replace("https://genius.com/", "")
                      .str.replace("-lyrics", "")
                      .str.replace("-", " ")
                      .apply(lambda x: tokenizer.tokenize(x)))

    urls["song"] = (urls["artist"].apply(lambda x: x[1:])
                    .apply(lambda x: " ".join(x)))
    urls["artist"] = urls["artist"].apply(lambda x: x[0])

    urls_ag = urls.groupby("artist")["song"].apply(list).reset_index()
    urls_ag["song"] = urls_ag["song"].apply(lambda x: [w for w in x if w != ""])
    urls_ag = urls_ag.loc[(urls_ag["song"].str.len() > 1) & (urls_ag["artist"].str.len() > 1), :]
    urls_ag = urls_ag.reset_index(drop=True)

    # Specific corrections
    urls_ag.loc[urls_ag["artist"] == "Bap", "artist"] = "Bap xl"
    bl = urls_ag.loc[urls_ag["artist"] == "Black", ["artist", "song"]].explode("song")
    bl.loc[bl["song"].str.startswith("m"), "artist"] = "Black M"
    bl.loc[bl["song"].str.startswith("kent"), "artist"] = "Black kent"
    bl.loc[bl["song"].str.startswith("kappa"), "artist"] = "Black kappa"
    bl = bl.groupby("artist")["song"].apply(list).reset_index()
    urls_ag = urls_ag.loc[urls_ag["artist"] != "Black", :]
    urls_ag = pd.concat([urls_ag, bl]).reset_index(drop=True).sort_index()
    return urls_ag


urlsag = clean_list_url(url_file)

os.chdir("data/")
for art, songs in zip(urlsag["artist"], urlsag["song"]):
    artist = genius.search_artist(art, max_songs=1, get_full_info=False)
    time.sleep(2)
    for song in songs:
        song = genius.search_song(song, artist.name, get_full_info=False)
        if song:
            song.save_lyrics()
        time.sleep(3)

# Df with artist and song
artists_dict = defaultdict(dict)
files_names = glob.glob("*.json")

for file_name in files_names:
    with open(file_name) as f:
        j = json.loads(f.read())
        artist_name = j["primary_artist"]["name"]
        artists_dict[artist_name][j["title"]] = j["lyrics"]

df = pd.DataFrame.from_dict(artists_dict, orient="columns").stack()
df = df.reset_index()
df.columns = ["song", "artist", "lyrics"]
print(df.shape)  # 470 chansons
df.to_csv("artist_dict.csv", encoding="utf-8", index=False)
