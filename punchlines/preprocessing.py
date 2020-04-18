# Preprocessing and exploratory analysis

# module
import numpy as np
import csv
import pandas as pd
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer, PorterStemmer, PortugueseStemmer, SpanishStemmer, GermanStemmer
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
from langdetect import detect

stem = dict()
stem["fr"] = FrenchStemmer()
stem["en"] = PorterStemmer()
stem["pt"] = PortugueseStemmer()
stem["es"] = SpanishStemmer()
stem["de"] = GermanStemmer()

tokenizer = nltk.RegexpTokenizer(r'[a-zA-Zéè\-\_]+')

# Import
lyrics = pd.read_csv("data/artist_dict.csv")
np.random.seed(123)
lyrics["lang"] = lyrics["lyrics"].apply(detect)


# Define stop words
languages = {'en': 'english', 'de': 'german', 'fr': 'french', 'es': 'spanish',
             'it': 'italian', 'pt': 'portuguese'}
stop_w_lan = {lan: set(stopwords.words(languages[lan])) for lan in languages.keys()}
# update stop words
stop_w_lan["en"].update(["songtext", "intro"])
stop_w_lan["fr"].update(["couplet", "refrain", "intro"])
stop_w_lan["es"].update(["refren"])


def preprocess(df):
    """Function to preprocess DataFrame containing lyrics by tokenizing, stemming,
    calculating vocabulary size, frequency of words, and taking window around word
    of interest
    :param df: pandas DataFrame, containing columns artist, song and lyrics
    :return : pandas DataFrame, with additional columns containing preprocessed info
    """

    df["tokens"] = df["lyrics"].str.lower().apply(tokenizer.tokenize)

    # Remove stop words by language
    def remove_stop_w(temp, st):
        temp = temp.apply(lambda x: [w for w in x if w not in st])
        return temp
    df["tokens_fil"] = df["tokens"]
    for i in stop_w_lan.keys():
        df.loc[df["lang"] == i, "tokens_fil"] = \
            remove_stop_w(df.loc[df["lang"] == i, "tokens"], stop_w_lan[i])

    # Stem by language
    def stem_by_lang(temp, stemmer):
        temp = temp.apply(lambda x: [stemmer.stem(w) for w in x])
        return temp

    df["tokens_fil_stem"] = df["tokens_fil"]
    for i in stem.keys():
        df.loc[df["lang"] == i, "tokens_fil_stem"] = \
            stem_by_lang(df.loc[df["lang"] == i, "tokens_fil"], stem[i])
    df = df.drop(columns=["lyrics"])
    df["tengrams"] = df["tokens_fil_stem"].apply(lambda x: list(nltk.ngrams(x, 30)))
    df["tengrams_no_stem"] = df["tokens_fil"].apply(lambda x: list(nltk.ngrams(x, 30)))

    r = re.compile(r"punch[a-zA-Z\-_]*")
    df["top_punch"] = df["tokens_fil_stem"].apply(lambda x: r.findall(" ".join(x)))
    print("Found word starting with punch in :{} songs/{} songs"
          .format((df["top_punch"].str.len() > 0).sum(), df.shape[0]))

    set_punch = set(df["top_punch"].explode())
    set_punch = {x for x in set_punch if pd.notna(x)}
    df["tengrams_fil"] = df["tengrams"].apply(lambda x: [w for w in x if set(w).intersection(set_punch)])
    df = df.loc[df["tengrams_fil"].str.len() > 0, :]

    df["tengrams_no_stem_fil"] = df["tengrams_no_stem"]\
        .apply(lambda x: [w for w in x if set(w).intersection(set_punch)])
    df = df.loc[df["tengrams_no_stem_fil"].str.len() > 0, :]

    df["vocab_size"] = df["tokens_fil_stem"].apply(lambda x: len(x))
    df["freq_dist"] = df["tokens_fil_stem"].apply(lambda x: nltk.FreqDist(x))
    df["freq_dist"] = df["freq_dist"].apply(lambda x: x.most_common(20))
    df["phrase_fil_stem"] = df["tengrams_fil"].apply(lambda x: [" ".join(w) for w in x])
    df["phrase_fil"] = df["tengrams_no_stem_fil"].apply(lambda x: [" ".join(w) for w in x])

    # keep only phrase where punch is beginning and end to get window

    exp_df = df.loc[:, ["song", "artist", "lang", "tengrams_fil", "phrase_fil_stem",
                        "phrase_fil", "vocab_size", "freq_dist"]]
    exp_df = exp_df.explode("phrase_fil")  # Because with stem difficult to interpret
    exp_df = exp_df.loc[(exp_df["phrase_fil"].str.startswith("punch")) |
                        (exp_df["phrase_fil"].str.contains(r"punch[a-zA-Z\-\_]*$")), :]

    exp_df.to_csv("data/artist_processed.csv", encoding="utf-8", index=False, sep=";", quoting=csv.QUOTE_ALL)
    return exp_df, set_punch


def save_word_cloud(text, masque, file_name, background_color="white"):
    mask_coloring = np.array(Image.open(str(masque)))
    wc = WordCloud(background_color=background_color, max_words=300,
                   stopwords=punch, mask=mask_coloring, random_state=123,
                   max_font_size=50, min_font_size=1)
    wc = wc.generate(text)
    image_colors = ImageColorGenerator(mask_coloring)
    wc = wc.recolor(color_func=image_colors)
    wc.to_file(file_name)


output, punch = preprocess(lyrics)

for lan in output["lang"].unique():
    window = " ".join(output.loc[output["lang"] == lan, "phrase_fil"])
    print("Language {} : count words : {}".format(lan, len(window)))
    save_word_cloud(window, "data/poing.jpg", "output/" + lan + "_songs.png")
