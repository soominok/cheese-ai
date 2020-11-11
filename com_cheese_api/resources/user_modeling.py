# import pandas as pd

# class UserModel:
#     def __init__(self):
#         self.user = None
#         self.user_top = None

#     def userFile(self):
#         self.user = pd.read_csv("com_cheese_api/resources/data/user_dataset.csv")
#         print(f'##### user 원본 데이터 출력#####')
#         print(self.user)

#         self.user['cheese_id'] = self.user['cheese_id'].str.replace('p', '')
#         self.user['cheese_one_price'] = self.user['cheese_one_price'].str.replace(',', '')
#         self.user['cheese_one_price'] = self.user['cheese_one_price'].str.replace('원', '')
#         self.user = self.user.astype({'cheese_one_price': int})
#         print(f'##### 치즈 id와 치즈 가격 형식 변환 #####')
#         print(self.user)

#     def peakData(self):
#         self.user_top = self.user[(self.user['user_age'] == 30) | (self.user['user_age'] == 40) | (self.user['user_age'] == 20) | (self.user['user_age'] == 50)]
#         print(f'User Top age Counts')
#         print(self.user_top['user_age'].value_counts())
#         print(f'User Top cheese Code Counts')
#         print(self.user_top['cheese_code'].value_counts())
#     # def dataSplit(self):


# if __name__ == '__main__':
#     userModel = UserModel()
#     userModel.peakData()


import pandas as pd
import sklearn.svm as svm
import sklearn.metrics as mt
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


class UserModel:
    def __init__(self):
        self.X_user_train = None
        self.X_user_test = None
        self.y_user_train = None
        self.y_user_test = None
        self.X_user_train_std = None
        self.X_user_test_std = None
        self.X_user_train_norm = None
        self.X_user_test_norm = None


    def userData(self):
        self.X_user_train = pd.read_csv("resources/data/X_user_top_train.csv")
        self.X_user_test = pd.read_csv("resources/data/X_user_top_test.csv")
        self.y_user_train = pd.read_csv("resources/data/y_user_top_train.csv")
        self.y_user_test = pd.read_csv("resources/data/y_user_top_test.csv")

    # def stdScale(self):
    #     std = StandardScaler()
    #     self.X_user_train_std = std.fit_transform(self.X_user_train)
    #     self.X_user_test_std = std.fit_transform(self.X_user_test)
    #     print(self.X_user_train_std)

    # def normScale(self):
    #     norm = MinMaxScaler()
    #     norm.fit(self.X_user_train)
    #     self.X_user_train_norm = norm.transform(self.X_user_train)
    #     self.X_user_test_norm = norm.transform(self.X_user_test)
    #     print(self.X_user_train_norm)

    @staticmethod
    def build_model_1():
        model = Sequential()
        model.add(Dense(8, activation = 'tanh', input_dim=8))
        model.add(Dense(4, activation = 'tanh'))
        model.add(Dense(1, activation = 'softmax'))

        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
        YourModel = model.fit(X_user_train, y_user_train, epochs = 100, batch_size = 5, verbose = 1)
        kf = KFold(n_splits = 5)
        results = cross_val_score(YourModel, X_user_train, y_user_train, cv = kf)
        return model


# YourModel = KerasRegressor(build_fn = build_model, epochs = 100, batch_size = 3, verbose = 1)

# kf = KFold(n_splits = 5)

# results = cross_val_score(YourModel, X_user_train_std, y_user_train_ohe, cv = kf)

if __name__ == '__main__':
    userModel = UserModel()
    # userModel.stdScale()
    userModel.build_model_1() 