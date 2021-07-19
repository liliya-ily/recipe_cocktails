import os
import datetime
import csv
import logging
import random
import numpy as np
import pandas as pd
from pandas.core.common import flatten

from transformers import AutoTokenizer, AutoConfig, AutoModelForPreTraining, \
    GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
# from PIL import Image
from aiogram import Bot, Dispatcher, executor, types
import telegram
from dotenv import load_dotenv


import markups as nav
from mydataset import myDataset, SPECIAL_TOKENS, MAXLEN
from static_text import *

# Чтобы запустить токен, нужно скачать библиотеку load_dotenv, создать в папке проекта
# ситемный файл .env и сохранить там переменную TOKEN. Пример ниже:
# TOKEN = '4837j434fn7nv843cnm38'
load_dotenv()
TOKEN = os.getenv("TOKEN")
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

logging.basicConfig(filename='log.txt', level=logging.INFO)


# Загрузка готовых рецептов
df_result = pd.read_csv('ready_recipe.csv')
df_result.drop('Unnamed: 0', axis=1, inplace=True)

# Объявление модели для создания рецепта
MODEL = 'sberbank-ai/rugpt3small_based_on_gpt2'

tokenizer = GPT2Tokenizer.from_pretrained(
    MODEL)
tokenizer.add_special_tokens(SPECIAL_TOKENS)
device = torch.device('cuda')
config = AutoConfig.from_pretrained(MODEL,
                                    bos_token_id=tokenizer.bos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    sep_token_id=tokenizer.sep_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    output_hidden_states=False)

model = AutoModelForPreTraining.from_pretrained(MODEL, config=config)
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load('.\model\ALCO.pt',
                                map_location=torch.device(device)))
model.cuda()


# Объявление модели для создания названия коктейля
config = AutoConfig.from_pretrained(MODEL,
                                    bos_token_id=tokenizer.bos_token_id,
                                    eos_token_id=tokenizer.eos_token_id,
                                    sep_token_id=tokenizer.sep_token_id,
                                    pad_token_id=tokenizer.pad_token_id,
                                    output_hidden_states=False)
model_title = AutoModelForPreTraining.from_pretrained(MODEL, config=config)
model_title.resize_token_embeddings(len(tokenizer))
model_title.load_state_dict(torch.load(
    '.\model\ALCO_title.pt', map_location=torch.device(device)))
model_title.cuda()

# Метод для обработки данных, введённых пользователем. В нём все буквы меняются в нижний регистр,
# заменяются известные бренды на похожие ингредиенты, убираются лишние пробелы.
# Разбивается строка на список ингредиентов, с учёом, что пользователи могут ввести разные сепараторы
def create_keywords(text_user):
    text_user = text_user.lower()
    text_user = text_user.replace('мартини', 'вермут',).replace(
        'швепс', 'тоник').replace(
        'пепси', 'кола').replace(
        'пэпси', 'кола')
    text_user = text_user.rstrip()
    text_user = text_user.lstrip()
    separators = [', ', ',', '. ', '.', ' ', '; ', ';']
    for separator in separators:
        if text_user.find(separator) != -1:
            text_user = text_user.split(separator)
            break
    if type(text_user) == str:
        text_user = [text_user]
    return text_user

# Метод для генерации рецептов на основе списка ингредиенты
def generate_recipe(text_user, kw):

    title = ""
    prompt = SPECIAL_TOKENS['bos_token'] + title + \
        SPECIAL_TOKENS['sep_token'] + kw + SPECIAL_TOKENS['sep_token']

    generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    device = torch.device("cuda")
    generated = generated.to(device)

    model.eval()

    return model.generate(generated,
                        do_sample=True,
                        min_length=50,
                        max_length=MAXLEN,
                        top_k=30,
                        top_p=0.7,
                        temperature=0.9,
                        repetition_penalty=2.0,
                        num_return_sequences=100
                        )

# Метод для выбора наиболее подходящих рецептов
def get_best_recipe(sample_outputs, title, keywords):

    recipes = []

    for i, sample_output in enumerate(sample_outputs):
        text = tokenizer.decode(sample_output, skip_special_tokens=True)
        a = len(title) + len(','.join(keywords))
        recipes.append("{}".format(text[a:]))

    actual_recipe_dict = {}

    for recipe in recipes:
        for i in keywords:
            if recipe.find(i.lower()[:-2]) != -1:
                if recipe in actual_recipe_dict:
                    actual_recipe_dict[recipe] += 1
                else:
                    actual_recipe_dict[recipe] = 1

    return sorted(actual_recipe_dict.items(), key=lambda x: -x[1])

# Метод для генерации названия рецептов на основе выбранного сгенерированного
# рецепта
def generate_title(keywords, recipe, kw):

    prompt = SPECIAL_TOKENS['bos_token'] + recipe + \
        SPECIAL_TOKENS['sep_token'] + kw + SPECIAL_TOKENS['sep_token']

    generated_title = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    device = torch.device("cuda")
    generated_title = generated_title.to(device)

    model_title.eval()

    return model_title.generate(generated_title,
                                do_sample=True,
                                min_length=50,
                                max_length=MAXLEN,
                                top_k=30,
                                top_p=0.7,
                                temperature=0.93,
                                repetition_penalty=2.0,
                                num_return_sequences=100
                                )

# Метод для выбора названия рецептов 
def get_random_title(sample_outputs_title, recipe, keywords):

    titles = []

    for i, sample_output in enumerate(sample_outputs_title):
        text = tokenizer.decode(sample_output, skip_special_tokens=True)
        a = len(recipe) + len(','.join(keywords))
        titles.append("{}".format(text[a:]))
        
    set_title = set(titles)
    
    random_titles = []
    for title in set_title:
        if len(title.split()) > 1:
            random_titles.append(title)
    return random.choices(random_titles, k=1)

# Метод для выбора наиболее подходящего существующего рецепта
# на основе списка ингредиентов
def get_ready_recipe(keywords, df_result=df_result):

    ready_recipe_dict = {}

    for index, row in df_result.iterrows():
        for keyword in keywords:
            if row.ingridients.find(keyword) != -1:
                if index in ready_recipe_dict:
                    ready_recipe_dict[index] += 1
                else:
                    ready_recipe_dict[index] = 1

    if ready_recipe_dict:
        max_ready_dict = max(ready_recipe_dict.values())
        ready_recipe = []
        for k, v in ready_recipe_dict.items():
            if v == max_ready_dict:
                ready_recipe.append(
                    [df_result["name"][k], df_result["recipe"][k]])
        return random.choices(ready_recipe, k=1)[0]
    else:
        return ['None']


# Метод для выдачи сгенерированного рецепта, названия для этого рецепта
# и существующего классического рецепта, содержащего введённые ингредиенты
def get_text(text_user):

    keywords = create_keywords(text_user)

    kw = myDataset.join_keywords(keywords, randomize=False)

    sample_outputs_recipe = generate_recipe(text_user, kw)

    sorted_tuple = get_best_recipe(sample_outputs_recipe, '', keywords)

    sample_outputs_title = generate_title(keywords, sorted_tuple[0][0], kw)
    recipe_title = get_random_title(
        sample_outputs_title, sorted_tuple[0][0], keywords)
    recipe = sorted_tuple[0][0].replace(' л ', ' мл ')

    ready_recipe = get_ready_recipe(keywords)
    if ('пиво' in keywords) and (recipe.find('пишо') != -1):
        recipe = recipe.replace('пишо', 'пиво')

    if len(ready_recipe) == 1:
        output = (
            f'<i>Рецепт от нашего бармена: </i>\n\n<b>{recipe_title[0]}</b>\
                \n\n {recipe}\n\n <i>Классический рецепт с такими игредиентами отсутствует</i>')
    else:
        output = (
            f'<i>Рецепт от нашего бармена: </i> <b>\n\n{recipe_title[0]}</b>\
                \n\n {recipe}\n\n <i>Классический рецепт:</i> \n\n <b>{ready_recipe[0]}</b> \n\n {ready_recipe[1]} ')
    logging.info(output)
    return output


# Команда для начала работы с ботом
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    user_name = message.from_user.first_name
    user_id = message.from_user.id
    text = HELLO_TEXT % user_name
    logging.info(
        f'First start from user_name = {user_name}, user_id = {user_id}')
    await message.reply(text)

# Проверяем входящие данные на НЕсоответствие типу "photo"
@dp.message_handler(content_types=NON_TARGET_CONTENT_TYPES)
async def handle_docs_photo(message):
    user_name = message.from_user.first_name
    text = NON_TARGET_TEXT % user_name
    await message.reply(text)

# Проверяем входящие данные на соответствие типу "text"
@dp.message_handler(content_types=['text'])
async def handle_docs_photo(message: types.Message):
    user_id = message.from_user.id
    if message.text == 'FAQ':
        i = open("input\shpargalka.png", 'rb')
        await bot.send_photo(user_id, i)
        await bot.send_message(message.from_user.id, 'Надеюсь, так стало понятнее. Введи ингредиенты через запятую!')

    elif message.text == 'Начать сначала':
        await bot.send_message(message.from_user.id, 'Введи ингредиенты через запятую:')

    elif message.text == 'другое':
        await bot.send_message(message.from_user.id, '➡️ Другое', reply_markup=nav.otherMenu)

    else:
        chat_id = message.chat.id

        user_name = message.from_user.first_name
        user_id = message.from_user.id
        message_id = message.message_id

        text = WAITING_TEXT

        logging.info(f'{user_name, user_id} is knocking to our bot')

        await bot.send_message(chat_id, text)
        p = open(".\input\8oVc.gif", 'rb')
        await bot.send_animation(chat_id, p)

        input_text = message.text
        logging.info(f'{user_name, user_id} send this text:{input_text}')
        output_text = get_text(input_text)

        await bot.send_message(chat_id, output_text, reply_markup=nav.mainMenu, parse_mode='html')

# Метод для вывода кнопок в телеграме
@dp.message_handler()
async def bot_message(message: types.Message):
    chat_id = message.chat.id
    if message.text == 'FAQ':
        i = open("input\shpargalka.png", 'rb')
        await bot.send_photo(chat_id, i)
        await bot.send_message(message.from_user.id, 'Надеюсь, так стало понятнее. Введи ингредиенты через запятую!')

    elif message.text == 'Начать сначала':
        await bot.send_message(message.from_user.id, 'Введи ингредиенты через запятую:')

    elif message.text == '➡️ Другое':
        await bot.send_message(message.from_user.id, '➡️ Другое', reply_markup=nav.otherMenu)


# Запуск бота в режиме длительного опроса
if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
