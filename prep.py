# Libraries
from gensim.similarities import MatrixSimilarity
from TurkishStemmer import TurkishStemmer
from gensim import corpora

import pandas as pd
import gensim
import pprint
import string
import spacy
import re

NUMBERS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]

spacy_nlp = spacy.blank('tr')
stop_words = list(spacy_nlp.Defaults.stop_words)
pp = pprint.PrettyPrinter(indent=4)
punctuations = string.punctuation
stemmer = TurkishStemmer()


def filter(query):
    # Remove puncuations
    query = re.sub(r'[^\w\s]', ' ', query)

    # Remove extra spaces
    query = re.sub(' +', ' ', query)

    # Get tokens
    tokens = spacy_nlp(query)

    # Lower, strip and lemmatize
    tokens = [word.text.lower().strip() for word in tokens]

    # Remove stopwords, and exclude words less than 2 characters
    tokens = [word for word in tokens if
              (word not in stop_words) and
              (word not in punctuations) and
              (word in NUMBERS or len(word) >= 2) and
              (word != "yok") and
              (word != "nan")]

    # Stem tokens - Does not works well :(
    # tokens = [stemmer.stem(word) for word in tokens]

    return tokens


def preprocess(df):
    used = ['STOK_ADI', 'ANA_KATEGORI_ADI', 'WEB_ANA_KATEGORI_TANIMI',
            'KATEGORI_ADI', 'ALT_KATEGORI_ADI', 'MARKA_ADI', 'RENK_ADI',
            'BEDEN', 'TEKSTIL_URUN_GRUBU_ADI', 'TEKSTIL_CINSIYET',
            'TEKSTIL_ADET_ADI', 'MALZEME_UZUN_ACIKLAMA', 'ASKI_DURUMU',
            'EN_STOK_ADI', 'ANALIZ_KATEGORISI', 'WEB_BIRINCI_KATEGORI_TANIMI',
            'WEB_IKINCI_KATEGORI_TANIMI', 'WEB_UCUNCU_KATEGORI_TANIMI',
            'MAGAZA_KONUMU']  # STOK_KODU Eklenecek buraya

    df = df[used]

    df["TEKSTIL_CINSIYET"] = df["TEKSTIL_CINSIYET"].replace(
        [1044, 1045, 1046], ["unisex", "kız", "erkek"])

    # Convert every column to string since we are dealing with NLP task
    df.fillna("yok", inplace=True)
    for feature in df.columns:
        df[feature] = df[feature].astype(str)

    for j, feature in enumerate(df.columns):
        if not j:
            df["final_tokenized"] = df[feature].map(lambda x: filter(x))
        else:
            df["final_tokenized"] += df[feature].map(lambda x: filter(x))

    return df


def main():
    df = pd.read_excel("./dataset/ebebek.xlsx")
    df = preprocess(df)

    keywords = df["final_tokenized"]

    # creating term dictionary
    dictionary = corpora.Dictionary(keywords)
    dictionary.save("./models/dictionary")
    print("Dictionary saved!")

    # creating corpus
    corpus = [dictionary.doc2bow(desc) for desc in keywords]

    # Creating TFIDF and LSI models
    ebebek_tfidf_model = gensim.models.TfidfModel(
        corpus, id2word=dictionary, smartirs="npu")
    ebebek_lsi_model = gensim.models.LsiModel(
        ebebek_tfidf_model[corpus], id2word=dictionary, num_topics=900)  # num_topic 900

    # Saving model corpus
    gensim.corpora.MmCorpus.serialize(
        './models/tfidf/ebebek_tfidf_corpus', ebebek_tfidf_model[corpus])
    gensim.corpora.MmCorpus.serialize(
        './models/lsi/ebebek_lsi_corpus', ebebek_lsi_model[ebebek_tfidf_model[corpus]])

    print("Model corpus saved!")

    # Saving TFIDF and LSI models
    ebebek_tfidf_model.save("./models/tfidf/ebebek_tfidf")
    ebebek_lsi_model.save("./models/lsi/ebebek_lsi")

    ebebek_tfidf_corpus = gensim.corpora.MmCorpus(
        './models/tfidf/ebebek_tfidf_corpus')
    ebebek_lsi_corpus = gensim.corpora.MmCorpus(
        './models/lsi/ebebek_lsi_corpus')

    malzeme_index = MatrixSimilarity(
        ebebek_lsi_corpus, num_features=ebebek_lsi_corpus.num_terms)
    malzeme_index.save("./models/malzeme_index")
    print("Malzeme index saved!")
    

if __name__ == "__main__":
    main()
