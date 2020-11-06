import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

import os

users = pd.read_csv("com_cheese_api/resources/data/users.csv")
# print(users)

cheeses = pd.read_csv("com_cheese_api/resources/data/cheese_data.csv")

def category_Count():
    sub_size = users['buy_count'].groupby(users['sub1_category']).sum().reset_index(name='sub1_counts')
    sub_size['sub1_rank'] = sub_size['sub1_counts'].rank(ascending=False)
    item_count_plot = sns.barplot (x = 'sub1_category', y = 'sub1_counts', data = sub_size)
    # plt.show()
    # print(sub_size)
    return sub_size

#category_Count()

def item_Count():
    # sub1_category에서의 빈도 추출
    category_count = category_Count()

    # sub2_category에서의 빈도 추출
    item_size = users['buy_count'].groupby([users['sub1_category'],users['sub2_category']]).sum().reset_index(name='sub2_counts')
    
    #item_size['sub2_rank'] = item_size['sub2_counts'].rank(ascending=False, method="dense")
    item_size['sub2_rank'] = item_size['sub2_counts'].rank(ascending=False)

    # sub1 + sub2 rank 합치기
    category_item_rank = pd.merge(category_count, item_size, on = 'sub1_category', how = 'right')
    # print(category_item_rank)

    # 원래 데이터에 rank 주기 (중복에 대한 계산도 같이 해주기)
    user_item_rank = pd.merge(users, category_item_rank, on = 'sub2_category', how = 'left')
    print(user_item_rank)

    user_items_ranks = user_item_rank.drop(['sub1_category_y'], axis=1)
    users_item_data = user_items_ranks.rename(columns={'sub1_category_x': 'sub1_category'})

    # users_item_data.to_csv(os.path.join('com_cheese_api/resources/data', 'user_item_counts3.csv'), index=False)
    return users_item_data

# item_count()


def item_Change():

    users_item_data = item_Count()
    cheese_data = cheeses.rename(columns={'ranking': 'sub2_rank'})
    users_data = pd.merge(users_item_data, cheese_data, on = 'sub2_rank', how = 'left')
    user_datas = users_data.drop(['category_x', 'sub1_category', 'sub2_category', 'item_brand', 'sub1_counts', 'sub1_rank', 'sub2_counts', 'sub2_rank', 'content', 'buy_price', 'img'], axis=1)
    user_data = user_datas.drop(['Unnamed: 0', 'country', 'matching', 'matching.1', 'texture', 'types', 'category_y'], axis=1)
    user_data_fin = user_data.rename(columns={'brand': 'cheese_brand', 'name': 'cheese_name', 'price' : 'cheese_one_price'})
    print(user_data_fin)

    user_data_fin.to_csv(os.path.join('com_cheese_api/resources/data', 'user_data.csv'), index=False)
    # # print(users_item_data)
    # items_lists = users_item_data['sub2_category']
    # # for 

    # category_count = category_Count()
    # category_lists = np.array(category_count)
    # print('-' * 100)
    # print(category_lists)
    # for data_category_list in users['sub1_category']:
    #     for category_num in range(len(category_lists)):
    #         if data_category_list in category_lists[category_num]:
    #             print()
    #     print(category_list)
    #     if category_list in 

# item_Change()


def show_df():
    user_data = pd.read_csv('com_cheese_api/resources/data/user_data.csv')
    show_user = user_data.head(10)
    show_column = list(user_data)
    print(show_user)
    print(show_column)


show_df()


