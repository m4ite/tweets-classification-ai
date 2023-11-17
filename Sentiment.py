import pandas as pd
from nltk.stem import PorterStemmer
import re
import string
from sklearn.model_selection import train_test_split
import operator
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import load, dump
import matplotlib.pyplot as plt
import seaborn as sns


sentiment_mapping = {'Positive': 0, 'Negative': 1, "Neutral": 2}
data = pd.read_csv('twitter_training.csv').drop('2401', axis=1)


# Renomeando as colunas
data.columns=['game','sentiment','tweet']

# Removendo valores nulos da coluna tweets
data = data.dropna(subset=['tweet'])
data['sentiment'] = data['sentiment'].replace("Irrelevant", "Neutral")


# Pegando as palavras mais relevantes dos tweets
with open("stopWords.txt", "r") as arquivo:
    stopwords = arquivo.read()

def process_tweet(tweet):
    stemmer = PorterStemmer()

    tweet = tweet.lower() 
    
    # Tirar emojis
    tweet = re.sub(r'<3', '<heart>' , tweet) 
    tweet = re.sub(r"[8:=;]['`\-]?[)d]+" , '<smile>' , tweet)
    tweet = re.sub(r"[8:=;]['`\-]?\(+" , '<sadface>' , tweet) 
    tweet = re.sub(r"[8:=;]['`\-]?[\/|l*]" , '<neutralface>' , tweet) 
    tweet = re.sub(r"[8:=;]['`\-]?p+" , '<lolface>' , tweet)
    
    tweet = re.sub(r"(.)\1\1+" , r"\1\1" , tweet)  # deixa só 2 letras repetidas seguidas
    tweet = re.sub("<br /><br />" , " " , tweet) #remove quebra de linhas
    tweet = re.sub(r"[^a-z0-9<>]" , ' ' , tweet) # remove simbolos
    tweet = re.sub('<.*?>', '', tweet) # HTML tags
    tweet = re.sub('((www.[^s]+)|([^s]+.com)|(https?://[^s]+))','',tweet)
    tweet = re.sub('[0-9]+', '', tweet) # numeros
    tweet = re.sub(r'[^\w\s]', '', tweet) # caracteres especiais
    tweet = re.sub(r'/' , ' / ' , tweet)
    tweet = re.sub(r'http\S+', '', tweet) # URLs ou web links
    tweet = re.sub(r'@\S+', '', tweet) # meções
    tweet = re.sub(r'#\S+', '', tweet) # hashtags
    tweet = re.sub('[^a-zA-Z#]+', ' ', tweet) # pontuação
    tweet = tweet.replace(r'\S+@\S+', "") # emails
    
    
    # remove stop words
    res = ""
    words = tweet.split(" ")
    for word in words:
        if (word not in stopwords and word not in string.punctuation):  
                stem_word = stemmer.stem(word)  
                res = res + stem_word + " "

    return res


#frequencia de cada palavra
result ={}
def count_tweets(result, tweets, ys):
    for y, tweet in zip(ys, tweets):
        lista = process_tweet(tweet).split(" ")
        for word in lista:
            # pair = (word, y)
            pair = word
            
            if pair in result:
                result[pair] += 1

            else:
                result[pair] = 1

    return result


#cria uma coluna nova para cada palavra significativa, colocando 0 ou 1 se o tweet possuir a palavra
def CountVectorizer(tweets, sigPalavras):
    for tupla in sigPalavras:
        palavra = tupla[0]
        array = []
        for tweet in tweets:
            if(palavra in tweet):
                array.append(1)
            else:
                array.append(0)
        data[palavra] = array
                


def ShowMetrics():
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Neutral', 'Negative', 'Positive'], yticklabels=['Neutral', 'Negative', 'Positive'])
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    plt.title('Confusion Matrix')
    plt.show()
    
    
def CountVectorizerRecived(tweet, sigPalavras):
    for tupla in sigPalavras:
        palavra = tupla[0]
        array = []
        if(palavra in tweet):
            array.append(1)
        else:
            array.append(0)
   
    return [X_test.iloc[0]]


def Predict(recivedData):
    model = load("PredictSentiment.pkl")
    array = CountVectorizerRecived(process_tweet(recivedData), maisSignificativas)
    prediction = model.predict(array)
    return prediction


# cria uma nova coluna com os dados processados
data.loc[:, 'prep'] = data['tweet'].apply(process_tweet)
data.loc[:, 'label'] = data['sentiment'].map(sentiment_mapping)
freqs = count_tweets({}, data['tweet'], data['label'])
category_counts = data['sentiment'].value_counts()


# seleciona as palavras que mais aparecem nos tweets
sortedDict = sorted(freqs.items(), key=operator.itemgetter(1))
del sortedDict[-1]
maisSignificativas = sortedDict[-2700:]
               
               
CountVectorizer(data['prep'], maisSignificativas)

# treina o modelo
X = data.drop(['sentiment', 'tweet', 'prep', 'label'], axis=1)
Y =  data['label']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=150)
nb_classifier = MultinomialNB().fit(X_train,y_train)
pred = nb_classifier.predict(X_test)
print('\n\n\nAccuracy: {0: .2f} %'.format(accuracy_score(y_test, pred)*100))

dump(nb_classifier, "PredictSentiment.pkl")

ShowMetrics()
