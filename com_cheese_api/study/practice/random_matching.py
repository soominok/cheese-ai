import numpy as np
import pandas as pd

import os

def cheeseData():
    cheese_data = pd.read_csv("com_cheese_api/study/data/cheese_data.csv")
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
    
matching_kind()

#def make_random_matching():

###########################################
print('#' * 30)

# def dummy_matching():
#     cheese_matching = cheese.matching.str.split('\s*,\s*', expand = True).stack().str.get_dummies().sum(level=0)
#     print(cheese_matching)
#     cheese_matching.to_csv(os.path.join('com_cheese_api/study/data', 'cheese_matching.csv'), index=True, encoding='utf-8-sig')

# dummy_matching()



def matching_list():
    cheese = cheeseData()
    matching_spacing = cheese.matching.str.split(',')
    # print(matching_spacing)
    cheese_matching = matching_spacing.apply(lambda x: pd.Series(x))

    # 컬럼을 행으로 변환 + matching 낱개만 가져오기
    cheese_row = cheese_matching.stack().reset_index(level = 1, drop = True).to_frame('matching_single')

    matching_df = cheese.merge(cheese_matching, left_index = True, right_index = True, how = 'left')
    # matching_list.to_csv(os.path.join('com_cheese_api/study/data', 'cheese_food.csv'), index=True, encoding='utf-8-sig')

    # 중복 행 찾아서 제거하기
    cheese_check_dup = cheese_row.duplicated()
    cheese_del_dup = cheese_row.drop_duplicates()

    # 리스트로 변환
    # cheese_list = list(np.array(cheese_del_dup[0].tolist()))
    cheese_list = list(cheese_del_dup['matching_single'])

    print(cheese_list)



matching_list()