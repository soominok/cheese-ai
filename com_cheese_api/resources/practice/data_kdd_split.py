import pandas as pd
from sklearn.model_selection import train_test_split

import os

users = pd.read_csv("com_cheese_api/resources/data/user_origin.csv")
user_name_change = users.rename(columns={'Unnamed: 0': 'index', 'cheese_code': 'item_code', 'cheese_name': 'item_name', 'cheese_add_name': 'item_add_name', 'cheese_brand': 'item_brand'})
user_column = user_name_change.drop(['item_name', 'item_add_name'], axis=1)
user_column.to_csv(os.path.join('com_cheese_api/resources/data', 'users.csv'), index=False)

#user_train, user_test = train_test_split(users, test_size=0.3, random_state = 32)

#user_train.to_csv(os.path.join('com_cheese_api/resources/data', 'user_train.csv'), index=False)
#user_test.to_csv(os.path.join('com_cheese_api/resources/data', 'user_test.csv'), index=False)