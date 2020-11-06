from com_cheese_api.ext.db import url, db, openSession, engine
from com_cheese_api.util.file import FileReader
from flask import request
from sqlalchemy import func
from sqlalchemy import and_, or_
from flask import Response, jsonify
from flask_restful import Resource, reqparse
from sklearn.ensemble import RandomForestClassifier # rforest
from sklearn.tree import DecisionTreeClassifier # dtree
from sklearn.ensemble import RandomForestClassifier # rforest
from sklearn.naive_bayes import GaussianNB # nb
from sklearn.neighbors import KNeighborsClassifier # knn
from sklearn.svm import SVC # svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold  # k value is understood as count
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import json
import os
import sys
from typing import List
from pathlib import Path

class CheeseDf:
    def __init__(self):
        self.fileReader = FileReader()
        self.data = os.path.join(os.path.abspath(os.path.dirname(__file__))+'/data')
        self.odf = None

    def new(self):
        this = self.fileReader
        cheese = 'cheese_data.csv'    
        this.cheese = self.new_model(cheese)
        print(this.cheese)
        # print(this)

        print(this.cheese.isnull().sum())

        this = CheeseDf.brand_merge_code(this)
        this = CheeseDf.ranking_ordinal(this)
        this = CheeseDf.cheese_texture_norminal(this)
        this = CheeseDf.types_norminal(this)
        this = CheeseDf.cheese_category_norminal(this)

        cheese_split = CheeseDf.df_split(this.cheese)

        # train, test 데이터#
        train = 'cheese_train.csv'
        test = 'cheese_test.csv'
        this = self.fileReader
        this.train = self.new_model(train) # payload
        this.test = self.new_model(test) # payload

        print(this)


        self.odf = pd.DataFrame(
            {
                'ranking' : this.train.ranking,
                'brand' : this.train.brand_code,
                'category' : this.train.category,
                'types': this.train.types,
                'matching': this.train.matching
            }
        )


        this.id = this.test['name']
        # print(f'Preprocessing Train Variable : {this.train.columns}')
        # print(f'Preprocessing Test Variable : {this.test.columns}')    
        this = CheeseDf.drop_feature(this, 'country')
        this = CheeseDf.drop_feature(this, 'price')
        this = CheeseDf.drop_feature(this, 'content')
        # print(f'Post-Drop Variable : {this.train.columns}')   



        # # print(f'Preprocessing Train Result: {this.train.head()}')
        # # print(f'Preprocessing Test Result: {this.test.head()}')
        # # print(f'Train NA Check: {this.train.isnull().sum()}')
        # # print(f'Test NA Check: {this.test.isnull().sum()}')    

        this.label = CheeseDf.create_label(this) # payload
        this.train = CheeseDf.create_train(this) # payload

        # # print(f'Train Variable: {this.train.columns}')
        # # print(f'Test Varibale: {this.test.columns}')
        # clf = RandomForestClassifier()
        # clf.fit(this.train, this.label)
        # prediction = clf.predict(this.test)

        # print(this)


        df = pd.DataFrame(

            {
                'texture': this.train.texture,
                'img' : this.train.img
                
            }

        )

        # print(self.odf)
        # print(df)
        sumdf = pd.concat([self.odf, df], axis=1)
        print(sumdf)
        print(sumdf.isnull().sum())
        print(list(sumdf))
        sumdf.to_csv(os.path.join('data', 'cheese_fin.csv'), index=False, encoding='utf-8-sig')
        return sumdf



    def new_model(self, payload) -> object:
        this = self.fileReader
        this.data = self.data
        this.fname = payload
        print(f'{self.data}')
        print(f'{this.fname}')
        return pd.read_csv(Path(self.data, this.fname)) 

    @staticmethod
    def create_train(this) -> object:
        return this.train.drop('name', axis = 1)
        

    @staticmethod
    def create_label(this) -> object:
        return this.train['name'] # Label is the answer.

    @staticmethod
    def drop_feature(this, feature) -> object:
        this.train = this.train.drop([feature], axis = 1)
        this.test = this.test.drop([feature], axis = 1)
        return this

    @staticmethod
    def brand_merge_code(this) -> object:
        brand_code = pd.read_csv("data/cheese_brand_code.csv")
        this.cheese = pd.merge(this.cheese, brand_code, left_on = 'brand', right_on='brand', how = 'left')
        return this

    @staticmethod
    def ranking_ordinal(this) -> object:
        return this

    @staticmethod
    def cheese_texture_norminal(this) -> object:
        this.cheese['texture'] = this.cheese['texture'].map({
            '후레쉬치즈': 1,
            '세미하드치즈': 2,
            '세미하드': 2,
            '하드치즈': 3,
            '소프트치즈': 4,
            '연성치즈': 5,
            '경성치즈': 6
        })
        return this

    @staticmethod
    def types_norminal(this) -> object:
        types_mapping = {'가공치즈':0, '자연치즈':1}
        this.cheese ['types'] = this.cheese['types'].map(types_mapping)
        this.cheese = this.cheese
        return this

    @staticmethod
    def cheese_category_norminal(this) -> object:
        category_map = {
            '모짜렐라': 1,
            '블루치즈': 2,
            '리코타': 3,
            '체다': 4,
            '파르미지아노 레지아노': 5,
            '고다': 6,
            '까망베르': 7,
            '브리': 8,
            '만체고': 9,
            '에멘탈': 10,
            '부라타': 11
        }
        this.cheese['category'] = this.cheese['category'].map(category_map)
        return this

    @staticmethod
    def df_split(data):
        cheese_train, cheese_test = train_test_split(data, test_size = 0.3, random_state = 32)
        cheese_train.to_csv(os.path.join('data', 'cheese_train.csv'), index=False)
        cheese_test.to_csv(os.path.join('data', 'cheese_test.csv'), index=False)       
        return cheese_train, cheese_test

# if __name__ == '__main__' :
#     df = CheeseDf()
#     df.new() 



'''
     cheese_id  ranking     brand  category  types  texture                                                img
0          33       33       샴피뇽         2      1        4  https://img-cf.kurly.com/shop/data/goods/15954...
1          48       48      푸글리제         3      1        1  https://img-cf.kurly.com/shop/data/goods/15319...
2          16       16      zott         1      1        1  https://img-cf.kurly.com/shop/data/goods/15266...
3          57       57    라 콘타디나         3      1        1  https://img-cf.kurly.com/shop/data/goods/15235...
4          47       47       란다나         6      1        2  https://img-cf.kurly.com/shop/data/goods/15777...
5          32       32       안젤로         2      1        2  https://img-cf.kurly.com/shop/data/goods/15107...
6          61       61       사토리         4      1        2  https://img-cf.kurly.com/shop/data/goods/15311...
7          54       54    퀘소로시난테         9      1        3  https://img-cf.kurly.com/shop/data/goods/15978...
8          49       49      베르기어         6      1        2  https://img-cf.kurly.com/shop/data/goods/15281...
9          69       69     그랑도르즈         7      1        4  https://img-cf.kurly.com/shop/data/goods/14775...
10         67       67       사토리         4      1        3  https://img-cf.kurly.com/shop/data/goods/15639...
[49 rows x 7 columns]
'''