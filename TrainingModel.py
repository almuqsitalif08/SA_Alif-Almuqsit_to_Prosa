import re
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss

# Baca dataset untuk Training dalam bentuk pandas
train_data = pd.read_csv('dataset/train_data_restaurant.tsv', sep='\t', names=['sentences', 'sentiment'])

# Visualisasi Persebaran Dataset pada Train Data
str_visualisasi = 'Visualisasi Persebaran Dataset pada Train Data'
print(str_visualisasi)
plt.figure(str_visualisasi)
sns.countplot(x='sentiment', data=train_data)
plt.show()
print(train_data["sentiment"].value_counts())

# Cleaning Data pada Data Training
# Saya ambil Stopword dari library yang telah disediakan oleh Satrawi
stop_factory = StopWordRemoverFactory()
# Saya tambah dengan beberapa kata yang terdapat dalam dataset dan tidak terlalu berpengaruh dalam
# pelatihan model sentimen analisis
stopword = stop_factory.get_stop_words() + ['dll', 'sy', 'trus', 'ny', 'has', 'been', 'di', 'sih', 'ke']
# Saya hapus beberapa kata dari stopword karena kata ini mengindikasikan sentimen negatif
stopword.remove('tidak')
stopword.remove('belum')
factory = StemmerFactory()
stemmer = factory.create_stemmer()
TEXT_CLEANING_RE = "@#=-_\S+|https?:\S+http?:S|{^A-Za-z}+"


def preprocessing(sentences):
    sentences = re.sub(TEXT_CLEANING_RE, '', str(sentences).lower()).strip()
    tokens = []
    for token in sentences.split():
        if token not in stopword:
            token = stemmer.stem(token)
            tokens.append(token)
    return " ".join(tokens)


train_data['clean_sentences'] = train_data.sentences.apply(lambda x: preprocessing(x))
print('Training Data telah berhasil dibersihkan')

# Training
# Saya menggunakan TFIDF Vektorisasi dan Arsitektur Logistic Regression
vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=10)
X_train = vectorizer.fit_transform(train_data.clean_sentences)
y_train = train_data['sentiment']

LR_ = LogisticRegression(C=3, solver='liblinear', max_iter=150).fit(X_train, y_train)
print('model telah berhasil dibuat')
# Didapatkan model dalam bentuk variabel LR_

# Preprocessing Test Dataset
# Baca Test dataset dalam bentuk pandas
test_data = pd.read_csv("dataset/test_data_restaurant.tsv", sep="\t", names=["sentences", "sentiment"])

# Visualisasi Persebaran Dataset pada Test Data
str_visualisasi = 'Visualisasi Persebaran Dataset pada Test Data'
print(str_visualisasi)
plt.figure(str_visualisasi)
sns.countplot(x='sentiment', data=test_data)
plt.show()
print(test_data["sentiment"].value_counts())

# Cleaning pada Test Data
test_data['clean_sentences'] = test_data.sentences.apply(lambda x: preprocessing(x))
print('test data telah berhasil dibersihkan')

# Testing Model menggunakan Test Data
X_test = vectorizer.transform(test_data.clean_sentences)
y_test = test_data['sentiment']

yhat = LR_.predict(X_test)
print('F1 Score : ', f1_score(y_test, yhat, average='weighted'))

yhat_prob = LR_.predict_log_proba(X_test)
print('Log Loss : ', log_loss(y_test, yhat_prob))
# Digunakan matriks F1 Score dan Loss sebagai penilaian terhadap model

# Menyimpan Model dan Vectorizer
file_model = 'LinearRegression.model'
pickle.dump(LR_, open(file_model, 'wb'))
print('model telah berhasil disimpan')

file_vectorizer = 'vectorizer.pickle'
pickle.dump(vectorizer, open(file_vectorizer, 'wb'))
print('vectorizer telah berhasil disimpan')
