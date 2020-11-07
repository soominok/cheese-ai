import pandas as pd
from com_cheese_api.util.file import FileReader

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from com_cheese_api.ext.db import url, db, openSession, engine
from time import sleep
import time
import random
from urllib.request import urlopen
from bs4 import BeautifulSoup
from bs4 import BeautifulSoup
from bs4.element import NavigableString

from sklearn.model_selection import train_test_split

import os
from pathlib import Path


# 데이터 정제 과정
class ReviewDf():
    def __init__(self):
        self.fileReader = FileReader()
        self.data = os.path.join(os.path.abspath(os.path.dirname(__file__))+'/data')
        self.odf = None

    def new(self):
        review = 'cheese_review_panda_2.csv'
        this = self.fileReader
        this.review = self.new_model(review)
        
        # this = ReviewDf.split_str_1(this.review, 'review_views', "[")
        # this = ReviewDf.split_str_0(this.review, 'review_views', "]")
        # this = ReviewDf.split_str_1(this.review, 'review_views', "'")
        # this = ReviewDf.split_str_1(this.review, 'review_detail', "\n")

        this = ReviewDf.cheese_category_norminal(this)

        this = ReviewDf.brand_merge_code(this)
        this = ReviewDf.change_column_name(this)
        print(this.review)

        review_split = ReviewDf.df_split(this.review)

        train = 'review_train.csv'
        test = 'review_test.csv'
        this = self.fileReader
        this.train = self.new_model(train) # payload
        this.test = self.new_model(test) # payload


        print(this.train)



        self.odf = pd.DataFrame(
            {
                'review_no' : this.train.review_no
            }
        )

        this = ReviewDf.drop_feature(this, 'review_date')

        # this.label = ReviewDf.create_label(this) # payload
        # this.train = ReviewDf.create_train(this) # payload

        df = pd.DataFrame(

            {
                'category': this.train.category,
                'brand' : this.train.brand_code,
                'cheese_name': this.train.product_name,
                'review_title': this.train.review_title,
                'review_detail': this.train.review_detail,
                'review_views': this.train.review_views
                
            }

        )

        # print(self.odf)
        # print(df)
        sumdf = pd.concat([self.odf, df], axis=1)
        print('#' * 10)
        print(sumdf)
        print(sumdf.isnull().sum())
        print(list(sumdf))
        sumdf.to_csv(os.path.join('com_cheese_api/study/data', 'review_fin.csv'), index=False, encoding='utf-8-sig')
        
        return sumdf


    def new_model(self, payload):
        this = self.fileReader
        this.data = self.data
        this.fname = payload
        print(f'{self.data}')
        print(f'{this.fname}')
        return pd.read_csv(Path(self.data, this.fname))

    # @staticmethod
    # def create_train(this) -> object:
    #     return this.train.drop('name', axis = 1)
        

    # @staticmethod
    # def create_label(this) -> object:
    #     return this.train['name'] # Label is the answer.

    @staticmethod
    def drop_feature(this, feature) -> object:
        this.train = this.train.drop([feature], axis = 1)
        this.test = this.test.drop([feature], axis = 1)
        return this

    @staticmethod
    def split_str_0(this, column, part):
        split = this[column].str.split(part)
        this[column] = split.str.get(0)
        return split

    @staticmethod
    def split_str_1(this, column, part):
        split = this[column].str.split(part)
        this[column] = split.str.get(1)
        return split

    @staticmethod
    def change_column_name(this):
        this.review = this.review.rename(columns={"Unnamed: 0": "review_no"})
        return this

    @staticmethod
    def cheese_category_norminal(this) -> object:
        category_map = {
            '크림': 0,
            '모짜렐라': 1,
            '고르곤졸라': 2,
            '리코타': 3,
            '체다': 4,
            '파마산': 5,
            '고다': 6,
            '까망베르': 7,
            '브리': 8,
            '만체고': 9,
            '에멘탈': 10,
            '부라타': 11
        }
        this.review['category'] = this.review['category'].map(category_map)
        return this


    @staticmethod
    def brand_merge_code(this) -> object:
        brand_code = pd.read_csv("com_cheese_api/study/data/cheese_brand_code.csv")
        this.review = pd.merge(this.review, brand_code, left_on = 'brand_name', right_on='brand', how = 'left')
        return this

#     # 결측값 제거 필요

    @staticmethod
    def df_split(data):
        review_train, review_test = train_test_split(data, test_size = 0.3, random_state = 32)
        review_train.to_csv(os.path.join('com_cheese_api/study/data', 'review_train.csv'), index=False)
        review_test.to_csv(os.path.join('com_cheese_api/study/data', 'review_test.csv'), index=False)       
        return review_train, review_test


#     # pandas로 가공한 데이터 csv 파일로 다시 저장하기
#     # 
#     # df.to_csv('cheese_review_panda.csv')
#     return df

if __name__ == "__main__":
    reviewDf = ReviewDf()
    reviewDf.new()

'''
        review_no  category  brand      cheese_name         review_title                                      review_detail  review_views
0          111         4     39          헤리티지 체다            아이가 좋아해서                                 자주 삽니다 ㅎㅎ 샐러드에도 제격!             4
1          153         4     39          헤리티지 체다                익숙한맛                       익숙해서 그런지 여러종류 시켰는데 체다가 제일맛있네요            15
2          614        10     14           에멘탈 치즈           예쁜 구멍이 뽕뽕   있는 향기도 맛도 질감도 쫀득한 맛있는 에멘탈치즈~~치즈는 사랑입니당♡.반은 먹어버...             7
3          313         5     39         사베치오 파마산                   오                                         깊은맛이 나요 맛나뇨            16
4          346         5     39         사베치오 파마산                맛있어요                                       깨끗한 맛이어서 좋아요.            20
..         ...       ...    ...              ...                  ...                                                ...           ...
695        892         6     24  110년 전통 고다치즈 3종           친구네서먹고 주문                           친구네서 먹었는데 아이도 넘좋아해서 구매...             2
696        310         5     39         사베치오 파마산                  굿굿                       그레이터가 아직 없어서 칼로 으깨 먹었는데 맛있었어요            12
697        901         8     32        덴마크 브리 치즈                  브리                                          좋아서 재구매했어요             0
698        555         7     32      덴마크 까망베르 치즈  쓴맛나는데 뭐가 맛있다는 거예요?   까망베르치즈 원래 이런 맛인가요? 어떻게 해먹어야 맛있는 건가요? 선물로 샀는데 맛...             9
699        727         2      7        고르곤졸라 피칸테                샐러드용           그린샐러드에 호두강정과 무화과 오렌지 리코타에 고르곤졸라까지 넣어서 냠냠.             7
'''