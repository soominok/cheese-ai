import numpy as np
import pandas as pd

import os

def cheeseData():
    cheese_data = pd.read_csv("com_cheese_api/resources/data/cheese_data.csv")
    return cheese_data

def matching_kind():
    cheese_data = cheeseData()
    # matchings = np.array(cheese_data['matching'].tolist())
    # # print(matchings)
    # matching_list = ','.join(matchings)
    # # print(matching_list)
    
    # matching_lists = matching_list.split(', , ')
    # # print(matching_lists)
    # print(type(matching_lists))

    # match_set = set(matching_lists)
    # match_only_list = list(match_set)
    # print('-' * 50)
    # print(match_only_list)

    matchings = np.array(cheese_data['matching'].tolist())
    # # print(matchings)
    # print(type(matchings))
    matching_list = ' '.join(matchings)
    print(matching_list)

# matching_kind()

# def dummy_matching():
#     cheese_matching = cheese.matching.str.split('\s*,\s*', expand = True).stack().str.get_dummies().sum(level=0)
#     print(cheese_matching)
#     cheese_matching.to_csv(os.path.join('com_cheese_api/resources/data', 'cheese_matching.csv'), index=True, encoding='utf-8-sig')

# dummy_matching()



def matching_list():
    cheese = cheeseData()
    matching_spacing = cheese.matching.str.split(',')
    # print(matching_spacing)
    cheese_matching = matching_spacing.apply(lambda x: pd.Series(x))

    # 컬럼을 행으로 변환 + matching 낱개만 가져오기
    cheese_only = cheese_matching.stack().reset_index(level = 1, drop = True).to_frame('matching_single')

    matching_list = cheese.merge(cheese_only, left_index = True, right_index = True, how = 'left')
    matching_list.to_csv(os.path.join('com_cheese_api/resources/data', 'cheese_food.csv'), index=True, encoding='utf-8-sig')
    
    print(cheese_only)
    print('-' * 30)
    print(cheese_matching)
    print('-' * 30)
    print(matching_list)



# matching_list()

# def matching_cheese():
#     cheese = cheeseData()
    