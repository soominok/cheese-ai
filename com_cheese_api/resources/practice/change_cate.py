import pandas as pd
import numpy as np

def change_category() :
    
    item_data = pd.read_csv("com_cheese_api/resources/data/users.csv")
    item_df = item_data.loc[:,['sub1_category']]
    item_lists = np.array(item_df['sub1_category'].tolist())
    #print(type(item_lists))

    rank_category = item_data['sub1_category'].value_counts()
    categories = np.array(rank_category.index).tolist()

    cheese_data = pd.read_csv("com_cheese_api/resources/data/cheese_data.csv")
    cheese_df = cheese_data.loc[:,['name']]
    cheese_lists = np.array(cheese_df['name']).tolist()
    #print(cheese_lists)

    match_value = {'영양제' : '모짜렐라', '생수' : '리코타', '닭고기류' : '블루치즈' , '냉동간편식' : '체다',\
                    '홍삼/인삼가공식품' : '파르미지아노 레지아노', '견과류' : '파르미지아노 레지아노', \
                        '포장반찬' : '고다', '건강진액' : '에멘탈', '건강보조식품' : '까망베르', '국산과일' : '부라타',\
                            '두유' : '만체고', '기능성음료' : '부라타', '축산선물세트' : '까망베르' }

    for item_list in range(len(item_lists)):
        item_df = item_data.replace({'sub1_category': match_value})
    print(item_df)

@staticmethod
def drop_feature(this, feature) -> object:
    # this.train = this.train.drop([feature], axis = 1)
    # this.test = this.test.drop([feature], axis = 1)
    return this

change_category()