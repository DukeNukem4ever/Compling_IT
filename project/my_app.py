#from flask import Flask, url_for, render_template, request
#import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import flask
import telebot
import conf
from telebot import types

WEBHOOK_URL_BASE = "https://{}:{}".format(conf.WEBHOOK_HOST, conf.WEBHOOK_PORT)
WEBHOOK_URL_PATH = "/{}/".format(conf.TOKEN)

bot = telebot.TeleBot(conf.TOKEN, threaded=False)

bot.remove_webhook()

bot.set_webhook(url=WEBHOOK_URL_BASE+WEBHOOK_URL_PATH)

app = flask.Flask(__name__)

count_vect = TfidfVectorizer()

answers = pd.read_excel('/home/DukeNukem4ever/mysite/answers_base.xlsx')
queries = pd.read_excel('/home/DukeNukem4ever/mysite/queries_base.xlsx')

train, test = train_test_split(queries, test_size=0.3)

new_train = pd.DataFrame()
new_train['Текст вопроса'] = pd.concat([train['Текст вопроса'],answers['Текст вопросов']],ignore_index = True)
new_train['Номер связки\n'] = pd.concat([train['Номер связки\n'], answers['Номер связки']], ignore_index=True)

X_train = count_vect.fit_transform(new_train['Текст вопроса'].values.astype('U'))
y_train = new_train['Номер связки\n']

X_test = count_vect.transform(test['Текст вопроса'].values.astype('U'))
y_test = test['Номер связки\n']

y_train = y_train.fillna(0)
y_test = y_test.fillna(0)

parameter = ['TFIDF']

tokenized_corpus = [doc.split(" ") for doc in new_train['Текст вопроса'].fillna('')]
bm25 = BM25Okapi(tokenized_corpus)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)
predicted = clf.predict(X_test)
acc = accuracy_score(y_test, predicted)

test['Predicted'] = predicted

test_csv = test.to_csv(index=False)

train_csv = train.to_csv(index=False)

matrix = count_vect.fit_transform(new_train['Текст вопроса'].fillna(' '))

def get_tfidf(text, vectorizer, m):
  text = text.lower()
  new_doc = vectorizer.transform([text]).toarray()
  new_m = m.dot(new_doc[0])
  return new_m

new_list = []
for d in new_train['Текст вопроса'].fillna(' '):
  new_list.append(get_tfidf(d,count_vect,matrix))
new_train['TFIDF'] = new_list

def get_bm25(query):
    query = query.lower()
    tokenized_query = query.split(" ")
    scores = bm25.get_scores(tokenized_query)
    return scores

other_list = []
for l in new_train['Текст вопроса'].fillna(' '):
  other_list.append(get_bm25(l))
new_train['BM25'] = other_list

def search_w2v(text):
  lemmas = text.split(' ')
  lemmas_vectors = np.zeros((len(lemmas), model.vector_size))
  vec = np.zeros((model.vector_size,))
  for idx, lemma in enumerate(lemmas):
    if lemma in model:
        lemmas_vectors[idx] = model[lemma]

  if lemmas_vectors.shape[0] != 0:
    vec = np.mean(lemmas_vectors, axis=0)
    return vec

def search_w2v_matrix(text):

  def normalize_vec(v):
     return v / np.sqrt(np.sum(v ** 2))

  def create_doc_matrix(text):
    lemmas = text.strip().split(' ')
    lemmas_vectors = np.zeros((len(lemmas), model.vector_size))
    vec = np.zeros((model.vector_size,))
    for idx, lemma in enumerate(lemmas):
        if lemma in model.wv:
            lemmas_vectors[idx] = normalize_vec(model.wv[lemma])

    return lemmas_vectors

  query_m = create_doc_matrix(text)
  docs_m = []

  for c in new_train['Текст вопросов']:
    docs_m.append(create_doc_matrix(c))

  def sim_search(docs, query, reduce_func=np.max, axis=0):
    sims = []
    for doc in docs:
        sim = doc.dot(query.T)
        sim = reduce_func(sim, axis=axis)
        sims.append(sim.sum())
    print(sims)
    return np.argmax(sims)

  return sim_search(docs_m, query_m)

n_conn = new_train['Номер связки\n'].tolist()

def ranging(vec):
  cnt = np.argmax(vec)
  number = n_conn[cnt]

  if int(number) == 0:
    result = 'Нулевой номер связки'
  else:
    answer = new_train[new_train['Номер связки\n'] == int(
        number)]['Текст вопроса'].tolist()
    result = 'Топ запрос: ' + str(
        number) + '. Ответ: ' + str(answer[0])[:350]+"..."
    return result

@bot.message_handler(commands=['start'])
def send_welcome(message):
    board = types.ReplyKeyboardMarkup(row_width=2)
    button1 = types.KeyboardButton('/TFIDF')
    button2 = types.KeyboardButton('/BM25')
    button3 = types.KeyboardButton('/Word2Vec_вектор')
    button4 = types.KeyboardButton('/Word2Vec_матрица')
    button5 = types.KeyboardButton('/Точность_модели')
    board.add(button1, button2, button3, button4, button5)#, button6, button7)
    bot.send_message(message.chat.id, "Здравствуйте! Это бот, посвящённый анализу таблицы с вопросами и ответами касательно коронавируса.")
    bot.send_message(message.chat.id, "Я могу провести поиск по тому или иному слову, составить его вектор или матрицу, а также найти по индексу самый близкий документ.")
    bot.send_message(message.chat.id, "Были взяты две таблицы - queries и answers. Они были разбиты на две выборки: 70% тренировочной выборки (все ответы на вопросы из answers и 70% вопросов из queries) и 30% тестовой выборки (оставшиеся 30% вопросов из queries).")
    bot.send_message(message.chat.id, "Введите /TFIDF, чтобы установить метод поиска как TFIDF (стоит по умолчанию).")
    bot.send_message(message.chat.id, "Введите /BM25, чтобы установить метод поиска как BM25.")
    bot.send_message(message.chat.id, "Введите /Word2Vec_вектор, чтобы установить метод поиска как Word2Vec и найти по нему вектор.")
    bot.send_message(message.chat.id, "Введите /Word2Vec_матрица, чтобы установить метод поиска как Word2Vec и найти по нему матрицу.")
    bot.send_message(message.chat.id, "После этого можете ввести любое слово/предложение/текст, чтобы я его обработал.")
    bot.send_message(message.chat.id, "Мой создатель - Орехов Артём Игоревич.", reply_markup=board)

@bot.message_handler(commands=['TFIDF'])
def send_welcome1(message):
    bot.send_message(message.chat.id, "Выбран метод TFIDF.")
    parameter.clear()
    parameter.append('TFIDF')

@bot.message_handler(commands=['BM25'])
def send_welcome2(message):
    bot.send_message(message.chat.id, "Выбран метод BM25.")
    parameter.clear()
    parameter.append('BM25')

@bot.message_handler(commands=['Word2Vec_вектор'])
def send_welcome3(message):
    bot.send_message(message.chat.id, "Выбран метод Word2Vec (вектор).")
    parameter.clear()
    parameter.append('Word2Vec_v')

@bot.message_handler(commands=['Word2Vec_матрица'])
def send_welcome4(message):
    bot.send_message(message.chat.id, "Выбран метод Word2Vec (матрица).")
    parameter.clear()
    parameter.append('Word2Vec_m')

@bot.message_handler(commands=['Точность_модели'])
def send_welcome5(message):
    bot.send_message(message.chat.id, "Точность данной модели равна {}".format(acc))
    bot.send_message(message.chat.id, "Для обучения модели использовались деревья решений (DecisionTreeClassifier).")

@bot.message_handler(func=lambda m: True)
def send_welcome6(message):
    bot.send_message(message.chat.id, "Осуществляется поиск...")
    if parameter == ['TFIDF']:
        tfidf = get_tfidf(str(message.text), count_vect, matrix)
        ranged = ranging(tfidf)
        bot.send_message(message.chat.id, "{}".format(ranged))
    elif parameter == ['BM25']:
        bm_25 = get_bm25(str(message.text))
        ranged = ranging(bm_25)
        bot.send_message(message.chat.id, "{}".format(ranged))
    elif parameter == ['Word2Vec_v']:
        bot.send_message(message.chat.id, "Упс! Что-то пошло не так...")
    elif parameter == ['Word2Vec_m']:
        bot.send_message(message.chat.id, "Упс! Что-то пошло не так...")
    else:
        bot.send_message(message.chat.id, "Упс! Что-то пошло не так...")


@app.route('/', methods=['GET', 'HEAD'])
def index():
    return 'ok'


@app.route(WEBHOOK_URL_PATH, methods=['POST'])
def webhook():
    if flask.request.headers.get('content-type') == 'application/json':
        json_string = flask.request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return ''
    else:
        flask.abort(403)

