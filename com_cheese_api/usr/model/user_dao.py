from com_cheese_api.usr.model.user_dto import UserDto
import numpy as np
import pandas as pd
from com_cheese_api.util.file import FileReader
from pathlib import Path
from com_cheese_api.ext.db import url, db, openSession, engine
from konlpy.tag import Okt
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sqlalchemy import func
from sqlalchemy.ext.declarative import declarative_base
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold  # k value is understood as count
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier # rforest
from sklearn.tree import DecisionTreeClassifier # dtree
from sklearn.ensemble import RandomForestClassifier # rforest
from sklearn.naive_bayes import GaussianNB # nb
from sklearn.neighbors import KNeighborsClassifier # knn
from sklearn.svm import SVC # svm


import os
import json


Session = openSession()
session = Session()

class UserDao(UserDto):
    # @classmethod
    # def bulk(cls, UserDf):
    #     userDf = UserDf()
    #     df = UserDf.new()
    #     print(df.head())
    #     session.bulk_insert_mappings(cls, df.to_dict(orient="records"))
    #     session.commit()
    #     session.close()
    @staticmethod
    def bulk():
        userDf = UserDf()
        df = userDf.new()
        print(df.head())
        session.bulk_insert_mappings(UserDto, df.to_dict(orient="records"))
        session.commit()
        session.close()

    @staticmethod
    def save(user):
        session.add(user)
        session.commit()

    @classmethod
    def update(cls, user):
        session.query(cls).filter(cls.user_no == user['user_no'])\
            .update({cls.password: user['password'],\
                cls.gender:user['gender'],\
                cls.age_group:user['age_group']})
        session.commit()

    @classmethod
    def delete(cls, user_no):
        data = cls.query.get(user_no)
        db.session.delete(data)
        db.session.commit()

    @classmethod
    def count(cls):
        return session.query(func.count(cls.user_no)).one()

    @classmethod
    def find_all(cls):
        # sql = cls.query
        # df = pd.read_sql(sql.statement, sql.session.bind)
        # return json.loads(df.to_json(orient='records'))
        return session.query(cls).all()

    '''
    SELECT * FROM users
    WHERE user_name LIKE a
    '''

    @classmethod
    def find_one(cls, user_id):
        return session.query(cls) \
            .filter(cls.user_id == user_id).one()

    # @classmethod
    # def find_by_name(cls, name):
    #     return session.query(cls).filter(cls.user_no.like(f'%{name}%')).all()

    @classmethod
    def find_users_in_category(cls, start, end):
        return session.query(cls) \
            .filter(cls.user_id.in_([start, end])).all()

    @classmethod
    def find_users_by_gender_and_age(cls, gender, age_group):
        return session.query(cls) \
            .filter(and_(cls.gender.like(gender),
            cls.age_group.like(f'{age_group}%'))).all()


    @classmethod
    def login(cls, user):
        return session.query(cls)\
            .filter(cls.user_id == user.user_id,
                cls.password == user.password).one()



if __name__ == '__main__':
    UserDao.bulk()