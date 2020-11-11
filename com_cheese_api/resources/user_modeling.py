# import numpy as np
# import pandas as pd
# from com_cheese_api.util.file import FileReader
# from pathlib import Path
# # from com_cheese_api.ext.db import url, db, openSession, engine
# # from konlpy.tag import Okt
# # from collections import Counter
# # from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import matplotlib.font_manager as fm
# import seaborn as sns
# from sqlalchemy import func
# from sqlalchemy.ext.declarative import declarative_base
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold  # k value is understood as count
# from sklearn.model_selection import cross_val_score
# from sklearn.ensemble import RandomForestClassifier # rforest
# from sklearn.tree import DecisionTreeClassifier # dtree
# from sklearn.ensemble import RandomForestClassifier # rforest
# from sklearn.naive_bayes import GaussianNB # nb
# from sklearn.neighbors import KNeighborsClassifier # knn
# from sklearn.svm import SVC # svm

# import os

# class UserModel:
#     def __init__(self):
#         self.fileReader = FileReader()
#         self.data = os.path.join(os.path.abspath(os.path.dirname(__file__))+'/data')
#         self.odf = None

#     def new(self):
#         user = 'user_dataset.csv'
#         this = self.fileReader
#         this.user = self.new_model(user) # payload

#         print('######## 데이터 확인 ##########')
#         print(this.user)


        
#         return this
#         # this.user = self.new_model(user)

#     def new_model(self, payload):
#         this = self.fileReader
#         this.data = self.data
#         this.fname = payload
#         print(f'{self.data}')
#         print(f'{this.fname}')
#         return pd.read_csv(Path(self.data, this.fname))


# if __name__ == '__main__':
#     userModel = UserModel()
#     userModel.new()

# class UserModel:
#     def __init__(self):
#         self.odf = None

#     @staticmethod
#     def readData():
#         users = pd.read_csv("com_cheese_api/resources/data/users.csv")

#         print(users)

# if __name__ == '__main__':
#     userModel = UserModel()
#     userModel.readData()


import pandas as pd

users = pd.read_csv("resources/data/user_dataset.csv")

# user_X_train_set = user_X_train.drop('user_id', axis = 1)
users['cheese_id'] = users['cheese_id'].str.replace('p','')
users['cheese_one_price'] = users['cheese_one_price'].str.replace(',', '')
users['cheese_one_price'] = users['cheese_one_price'].str.replace('원', '')
users = users.astype({'cheese_one_price': int})

# user_X_trian_set = user_X_train_set.astype({'cheese_one_price': int})