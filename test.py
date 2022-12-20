# Libraries
from TurkishStemmer import TurkishStemmer
from operator import itemgetter

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


def search_similar_products(search_term, dictionary, ebebek_tfidf_model, ebebek_lsi_model, malzeme_index, df, num_best=100):
    query_bow = dictionary.doc2bow(filter(search_term))
    query_tfidf = ebebek_tfidf_model[query_bow]
    query_lsi = ebebek_lsi_model[query_tfidf]

    malzeme_index.num_best = num_best

    malzeme_list = malzeme_index[query_lsi]

    malzeme_list.sort(key=itemgetter(1), reverse=True)
    malzeme_names = []

    for j, malzeme in enumerate(malzeme_list):

        malzeme_names.append(
            {
                'SEARCH_TERM': search_term,
                'STOK_ADI': df['STOK_ADI'][malzeme[0]],
                'MALZEME_UZUN_ACIKLAMA': df['MALZEME_UZUN_ACIKLAMA'][malzeme[0]],
                'RELEVANCE': round((malzeme[1] * 100), 2)
            }  # STOK_KODU Eklenecek

        )
        if j == (malzeme_index.num_best-1):
            break

    # STOK_KODU Eklenecek
    return pd.DataFrame(malzeme_names, columns=['SEARCH_TERM', 'STOK_ADI', 'MALZEME_UZUN_ACIKLAMA', 'RELEVANCE'])


def main():
    df = pd.read_excel("./dataset/ebebek.xlsx")

    used = ['STOK_ADI', 'ANA_KATEGORI_ADI', 'WEB_ANA_KATEGORI_TANIMI',
            'KATEGORI_ADI', 'ALT_KATEGORI_ADI', 'MARKA_ADI', 'RENK_ADI',
            'BEDEN', 'TEKSTIL_URUN_GRUBU_ADI', 'TEKSTIL_CINSIYET',
            'TEKSTIL_ADET_ADI', 'MALZEME_UZUN_ACIKLAMA', 'ASKI_DURUMU',
            'EN_STOK_ADI', 'ANALIZ_KATEGORISI', 'WEB_BIRINCI_KATEGORI_TANIMI',
            'WEB_IKINCI_KATEGORI_TANIMI', 'WEB_UCUNCU_KATEGORI_TANIMI',
            'MAGAZA_KONUMU']  # STOK_KODU Eklenecek buraya

    df = df[used]

    dictionary = gensim.corpora.Dictionary.load("./models/dictionary")
    malzeme_index = gensim.similarities.MatrixSimilarity.load(
        "./models/malzeme_index")

    # Load TFIDF and LSI models
    ebebek_tfidf_model = gensim.models.TfidfModel.load(
        "./models/tfidf/ebebek_tfidf")
    ebebek_lsi_model = gensim.models.LsiModel.load("./models/lsi/ebebek_lsi")

    similar_products = search_similar_products(
        "Kırmızı elbise", dictionary, ebebek_tfidf_model, ebebek_lsi_model, malzeme_index, df, 5)
    print(similar_products)


if __name__ == "__main__":
    main()
