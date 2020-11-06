import numpy as np
import pandas as pd
# from com_cheese_api.util.file import FileReader
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def make_wordcloud():

        item_data = pd.read_csv("com_cheese_api/resources/data/users.csv")
        item_df = item_data.loc[:,['cheese_name']]
        item_lists = np.array(item_df['cheese_name'].tolist())
                
        with open('com_cheese_api/user/data/stopword.txt', 'r') as file:
            lines = file.readlines()
            stop_str = ''.join(lines)
            stopword = stop_str.replace('\n', ' ')
        stopwords = stopword.split(' ')

        okt = Okt()

        sentences_tag = []
        
        #형태소 분석하여 리스트에 넣기
        for sentence in item_lists:
            morph = okt.pos(sentence)
            sentences_tag.append(morph)
            #print(morph)
            #print('-' * 30)
        
        #print(sentences_tag)
        #print('\n' * 3)
        
        noun_adj_list = []
        #명사와 형용사만 구분하여 이스트에 넣기
        for sentence1 in sentences_tag:
            for word, tag in sentence1:
                if word not in stopwords:
                    if tag in ['Noun']:
                        if len(word) >= 2:
                            noun_adj_list.append(word)
                        
        
        word_count_list = []
        #형태소별 count
        counts = Counter(noun_adj_list)
        tags = counts.most_common(100)
        word_count_list.append(tags)
        word_list = sum(word_count_list, [])
        print(word_list)
        print(type(word_list))
        
        
        # wordCloud 생성
        # 한글 깨지는 문제 해결하기 위해 font_path 지정
        wc = WordCloud(font_path='/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf', background_color='white', width=800, height=600)
        #print(dict(tags))
        cloud = wc.generate_from_frequencies(dict(tags))
        plt.figure(figsize=(10, 8))
        plt.axis('off')
        plt.imshow(cloud)
        return plt.show()

make_wordcloud()