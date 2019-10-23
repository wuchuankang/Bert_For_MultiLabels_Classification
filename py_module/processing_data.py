import pandas as pd
import torch as t
import numpy as np
from collections import defaultdict
from transformers import BertTokenizer


comment_train = pd.read_csv('./comment-classification/ai_challenger_sentiment_analysis_trainingset_20180816/sentiment_analysis_trainingset.csv')
comment_val = pd.read_csv('./comment-classification/ai_challenger_sentiment_analysis_validationset_20180816/sentiment_analysis_validationset.csv')
comment_test = pd.read_csv('./comment-classification/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv') 



columns = ['location_traffic_convenience',
       'location_distance_from_business_district', 'location_easy_to_find',
       'service_wait_time', 'service_waiters_attitude',
       'service_parking_convenience', 'service_serving_speed', 'price_level',
       'price_cost_effective', 'price_discount', 'environment_decoration',
       'environment_noise', 'environment_space', 'environment_cleaness',
       'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
       'others_overall_experience', 'others_willing_to_consume_again']

comment_train[columns] += 2
comment_val[columns] += 2

import re
def remove_special_token(text):
    text = re.sub('\n','',text)
    text = re.sub(r'[_"\-;；%()|+&=*%.。，！？￥,!?:#$@\[\]/]', '', text)
    text = re.sub(r'\'', '', text)
    text = re.sub(r'[a-z]', '', text)
    text = re.sub(r'[A-Z]', '', text)
    text = re.sub(r'[【】《》（）“”’‘：⋯<>『』、‿≖✧▶★⭐️☆˘д❖😄👍😱😋🐮😊]', '', text)
    text = re.sub(r'[\^￣—╭♡~～〜﹏≧▽≦〈〉…☁❤Ő௰⊙]', '', text)
    text = re.sub(r'[\x01\x02\x03\x04\x05\x06\x07\x08]', '', text)
    text = re.sub(r'[0-9①②③④⑤⑥⑦⑧⑨⑩๑•̀ㅂ•́]', '', text)
    text = re.sub(r'\r', '',text)
    text = re.sub(r'\\', '',text)
    text = re.sub(r' ', '',text)
    text = re.sub(r'\t', '',text)
    return text

# 去除正则化中的特殊字符
def remove_special_words(reviews):
    for i in range(len(reviews)):
        reviews[i] = remove_special_token(reviews[i])


remove_special_words(comment_train.content)
remove_special_words(comment_val.content)
remove_special_words(comment_test.content)

# 找出所有不在 vocab 中的字和表情
def out_of_vocab(content):
    unk_words = defaultdict(list)

    stics = []
    for i in range(len(content)):
        unk_count = 0
        for word in content[i]:
            if word not in tokenizer.vocab.keys():
                unk_count += 1
                unk_words[i].append(word)
        if unk_count > 0:
            stics += unk_words[i] 
    return set(stics)

# 建立 词典和分词对象
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

train_stop_words = out_of_vocab(comment_train.content)
val_stop_words = out_of_vocab(comment_val.content)
test_stop_words = out_of_vocab(comment_test.content)

# 删除不在字典中字符
def remove_oov(reviews, my_stop_words):
    for i in range(len(reviews)):
        reviews[i] = ''.join([word for word in reviews[i] if word not in my_stop_words])

remove_oov(comment_train.content, val_stop_words)
remove_oov(comment_val.content, val_stop_words)
remove_oov(comment_test.content, val_stop_words)


# 保存为csv文件，以备 Bert_For_MultiLabels_Classification.ipynb 使用
comment_train.to_csv('./my_data/sentiment_analysis_trainingset.csv', index=False)
comment_val.to_csv('./my_data/sentiment_analysis_validationset.csv', index=False)
comment_test.to_csv('./my_data/sentiment_analysis_testset.csv', index=False)
