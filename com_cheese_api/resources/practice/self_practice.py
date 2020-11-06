import numpy as np
import pandas as pd
from com_cheese_api.util.file import FileReader
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

from sklearn.model_selection import train_test_split

import os

class UserDf:
    def __init__(self):
        self.fileReader = FileReader()
        self.data = os.path.join(os.path.abspath(os.path.dirname(__file__))+'/data')
        self.odf = None

    #------------------------------------------ 데이터 셋 정제 ------------------------------------------#

    @staticmethod
    def userOrigin():
        user_data = pd.read_csv("com_cheese_api/resources/data/users.csv")
        print(user_data)
        return user_data

    @staticmethod
    def make_barplot(x_name, y_name, data_name):
        font_path = 'C:\\Windows\\Fonts\\NanumGothic.ttf'
        font_name1 = fm.FontProperties(fname = font_path).get_name()
        plt.rc('font', family = font_name1)
        plt.xticks(rotation = 45)
        sns.barplot(x = x_name, y = y_name, data = data_name)
        # plt.show()

    @staticmethod
    def category_Count():
        user_data = UserDf.userOrigin()
        sub_size = user_data['buy_count'].groupby(user_data['sub1_category']).sum().reset_index(name='sub1_counts')
        sub_size['sub1_rank'] = sub_size['sub1_counts'].rank(ascending=False)
        barplot = UserDf.make_barplot('sub1_category', 'sub1_counts', sub_size)
        return sub_size

    @staticmethod
    def cheeseData():
        cheese_data = pd.read_csv("com_cheese_api/resources/data/cheese_data.csv")
        return cheese_data

    @staticmethod
    def item_Count():
        user_data = UserDf.userOrigin()
        category_count = UserDf.category_Count()

        item_size = user_data['buy_count'].groupby([user_data['sub1_category'],user_data['sub2_category']]).sum().reset_index(name='sub2_counts')
        item_size['sub2_rank'] = item_size['sub2_counts'].rank(ascending=False, method="dense")

        category_item_rank = pd.merge(category_count, item_size, on = 'sub1_category', how = 'right')
        user_item_rank = pd.merge(user_data, category_item_rank, on = 'sub2_category', how = 'left')
        # print(user_item_rank)

        user_items_ranks = user_item_rank.drop(['sub1_category_y'], axis=1)
        users_item_data = user_items_ranks.rename(columns={'sub1_category_x': 'sub1_category'})
        # print(users_item_data)

        # users_item_data.to_csv(os.path.join('com_cheese_api/resources/data', 'user_item_counts3.csv'), index=False)
        return users_item_data

    @staticmethod
    def item_Change():
        cheese_data = UserDf.cheeseData()
        users_item_data = UserDf.item_Count()

        cheese_df = cheese_data.rename(columns={'ranking': 'sub2_rank'})
        user_cheese_merge = pd.merge(users_item_data, cheese_df, on = 'sub2_rank', how = 'left')
        user_data1 = user_cheese_merge.drop(['item_code', 'item_name', 'item_add_name', 'category_x', 'sub1_category', 'sub2_category', 'item_brand', 'sub1_counts', 'sub1_rank', 'sub2_counts', 'buy_price'], axis=1)
        user_data2 = user_data1.drop(['country', 'matching', 'matching.1', 'content', 'img'], axis=1)
        user_data_fin = user_data2.rename(columns={'Unnamed: 0_x': 'user_index', 'Unnamed: 0_y': 'cheese_code', 'brand': 'cheese_brand', 'name': 'cheese_name', 'price' : 'cheese_one_price', 'sub2_rank': 'cheese_rank', \
                                                        'category_y': 'cheese_category', 'texture': 'cheese_texture', 'types': 'cheese_types'})
        # print(list(users_cheese_merge))
        # print(user_data_fin)
        user_data_fin.to_csv(os.path.join('com_cheese_api/resources/data', 'user_data.csv'), index=False)
        return user_data_fin
    # item_Change()

    # 데이터 정제 끝난 User 데이터 셋!
    @staticmethod
    def userData():
        user_data = UserDf.item_Change()
        return user_data

    @staticmethod
    def show_User_Df():
        user_data = UserDf.userData()
        show_user = user_data.head(10)
        show_column = list(user_data)
        print(show_user)
        print(show_column)
        return user_data

    @staticmethod
    def data_split ():
        user_data = UserDf.userData()
        user_train, user_test = train_test_split(user_data, test_size=0.3, random_state = 32)
        user_train.to_csv(os.path.join('com_cheese_api/resources/data', 'user_train.csv'), index=False)
        user_test.to_csv(os.path.join('com_cheese_api/resources/data', 'user_test.csv'), index=False)
        return user_train, user_test



#######################
    def new_model(self, payload) -> object:
        this = self.fileReader
        this.data = self.data
        this.fname = payload
        print(f'{self.data}')
        print(f'{this.fname}')
        return pd.read_csv(Path(self.data, this.fname))

    @staticmethod
    def create_train(this):
        return this.train.drop('cheese_category', axis=1)

    @staticmethod
    def create_label(this):
        return this.train['cheese_category'] # Label = answer

    @staticmethod
    def drop_feature(this, feature):
        this.train = this.train.drop([feature], axis = 1)
        this.test = this.test.drop([feature], axis = 1)
        return this
        
    @staticmethod
    def gender_norminal(this):
        combine = [this.train, this.test]
        gender_mapping = {'male': 0, 'female': 1}
        for dataset in combine:
            dataset['user_gender'] = dataset['user_gender'].map(gender_mapping)
        this.train = this.train
        this.test = this.test
        return this


    # def user_Corr():
    #     user_data = UserDf.userData()
    #     userCorr = user 
    # def make_heatmap

if __name__ == '__main__':
    UserDf.show_User_Df()

#------------------------------------------ 데이터 탐색 & 시각화 ------------------------------------------#

    # def make_wordcloud(self):
    #     user_data = show_df()
    #     user_df = _data.loc[:,['cheese_name']]
    #     user_lists = np.array(user_df['cheese_name'].tolist())
                
    #     with open('com_cheese_api/user/data/stopword.txt', 'r') as file:
    #         lines = file.readlines()
    #         stop_str = ''.join(lines)
    #         stopword = stop_str.replace('\n', ' ')
    #     stopwords = stopword.split(' ')

    #     sentences_tag = []
        
    #     #형태소 분석하여 리스트에 넣기
    #     for sentence in _lists:
    #         morph = self.okt.pos(sentence)
    #         sentences_tag.append(morph)
    #         #print(morph)
    #         #print('-' * 30)
        
    #     #print(sentences_tag)
    #     #print('\n' * 3)
        
    #     noun_adj_list = []
    #     #명사와 형용사만 구분하여 이스트에 넣기
    #     for sentence1 in sentences_tag:
    #         for word, tag in sentence1:
    #             if word not in stopwords:
    #                 if tag in ['Noun']:
    #                     if len(word) >= 2:
    #                         noun_adj_list.append(word)
                        
        
    #     word_count_list = []
    #     #형태소별 count
    #     counts = Counter(noun_adj_list)
    #     tags = counts.most_common(100)
    #     word_count_list.append(tags)
    #     word_list = sum(word_count_list, [])
    #     print(word_list)
    #     print(type(word_list))
    
    #     # wordCloud 생성
    #     # 한글 깨지는 문제 해결하기 위해 font_path 지정
    #     wc = WordCloud(font_path='/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf', background_color='white', width=800, height=600)
    #     #print(dict(tags))
    #     cloud = wc.generate_from_frequencies(dict(tags))
    #     plt.figure(figsize=(10, 8))
    #     plt.axis('off')
    #     plt.imshow(cloud)
    #     return plt.show()

    

