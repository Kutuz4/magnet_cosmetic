import asyncio
from bs4 import BeautifulSoup as soup
import aiohttp
import json
import pandas as pd
import streamlit as st
import requests
import matplotlib.pyplot as plt
import time
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as Forest
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

cities_url_list = [55568, 56500, 93848, 79623, 106911, 57949, 131687]
cities = {key:value for key, value in zip(cities_url_list, ['Москва', 'Зеленоград', 'Коммунарка', 'Марушкино', 'Московский', 'Щербинка', 'п. Шишкин Лес'])}

def decoding(st):
    if isinstance(st, str):
        return st.encode('ascii').decode('unicode_escape')
    return st

async def decode_shop(jsn):
    tmp = dict()
    stops = set(['fShop', 'way', 'close', 'rating', 'target', 'city'])
    for key in jsn['shops'][0].keys():
        if key not in stops:
            tmp[key] = decoding(jsn['shops'][0][key])
    tmp['lat'] = tmp['coords']['lat']
    tmp['lng'] = tmp['coords']['lng']
    tmp['city'] = jsn['shops'][0]['city']
    return tmp

async def get(session, city_url, city_id):
    async with aiohttp.ClientSession() as session:
        async with session.get(f'https://magnitcosmetic.ru{city_url}') as response:
            res = await response.read()
            result = await parse_shops(res, city_id)
            return result
        
async def parse_shops(content, city_id):
    site = str(content)
    one = site.index('oneShopData')
    data = json.loads(site[one + 14: site.index('</script>', one)])
    data['shops'][0]['city'] = cities[city_id]
    return data

def parse_shops_magnet(id):
    url = f'https://magnitcosmetic.ru/shops/shop_list.php?city_id={id}'
    response = requests.get(url)
    if response.ok:
        shops_list = list(map(lambda x: x.find('a')['href'], soup(response.text, 'html.parser').find_all('div', {'class':'shops__address'})))
        return shops_list
    else:
        return False
    
async def parse():
    tasks = list()
    async with aiohttp.ClientSession() as session:
        for city_id in cities_url_list:
            for shop in parse_shops_magnet(city_id):
                tasks.append(get(session, shop, city_id))
        data = await asyncio.gather(*tasks)
        return data
    
async def gather_tasks(tasks):
    return await asyncio.gather(*tasks)

def get_data():
    tasks = [decode_shop(shop) for shop in asyncio.run(parse())]
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = asyncio.run(gather_tasks(tasks))
    return tasks

def main():
    tasks = get_data()
    df = pd.DataFrame(tasks).drop('weekend', axis=1)
    df['time'] = df['time'].apply(lambda x: '|'.join([x for i in range(7)]))
    df['shop'] = pd.Series(['Магнит-косметик' for i in range(len(df))], index=df.index)
    df['country'] = pd.Series(['Russia' for i in range(len(df))], index=df.index)
    return df

def agg_ml(dataframe, Model):
    X, y = dataframe[['lat', 'lng']], dataframe['city']
    encoder = {y.unique()[i]:i for i in range(len(y.unique()))}
    decoder = {encoder[key]:key for key in encoder.keys()}
    y = y.apply(lambda x: encoder[x])
    t = time.time()
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.35)
    model = Model()
    model.fit(train_X, train_y)
    t = time.time() - t
    st.write(f'Время обучения:{t}')
    scoref1 = f1_score(test_y, model.predict(test_X), average='macro')
    scorea1 = accuracy_score(test_y, model.predict(test_X))
    scoref0 = f1_score(train_y, model.predict(train_X), average='macro')
    scorea0 = accuracy_score(train_y, model.predict(train_X))
    st.write(f'Тренировочные метрики - Accuracy:{scorea0}, F1 score:{scoref0}')
    st.write(f'Тестовые метрики - Accuracy:{scorea1}, F1 score:{scoref1}')
    st.write('Посмотрим, как машина определила города из выборки')
    dataframe['predicted_city'] = pd.Series(model.predict(dataframe[['lat', 'lng']])).apply(lambda x: decoder[x])
    fig, axes = plt.subplots()
    for city in dataframe['predicted_city'].unique():
        cities = dataframe.query(f'predicted_city == "{city}"')
        axes.scatter(cities['lat'], cities['lng'], color=colors[city])
    axes.legend([city for city in dataframe['predicted_city'].unique()])
    st.write(fig)
    
if __name__ == '__main__':
    dataframe = main()
    columns = dataframe.columns
    st.title('Анализ адресов магнит-косметик')
    st.write('Заголовок таблицы данных:')
    st.dataframe(dataframe.head())
    st.write('Аналитика по городам: ')
    st.dataframe(dataframe.groupby('city')['id'].count())
    st.write(dataframe.groupby('city')['id'].count().plot(kind='barh').get_figure())
    colors = {x: [random.random(), random.random(), random.random()] for x in dataframe['city'].unique()}
    fig, axes = plt.subplots()
    for city in colors.keys():
        cities = dataframe.query(f'city == "{city}"')
        axes.scatter(cities['lat'], cities['lng'], color=colors[city])
    st.write(fig)
    st.write('Как мы видим, большая часть городов является ближайшим Подмосковьем')
    st.title('Машинное обучение')
    st.write('Попробуем применить модели машинного обучения для определения города по координатам')
    st.write('Случайный лес')
    agg_ml(dataframe, Forest)
    st.write('Наивный байесовский классификатор')
    agg_ml(dataframe, GaussianNB)
    st.write('Логистическая регрессия')
    agg_ml(dataframe, LogisticRegression)
    st.write('Как мы можем увидеть, логистическая регрессия справилась сильно хуже в условиях дисбаланса классов. Перезапустите страницу для повторного тестирования')
