import pandas as pd
import os

cheese = pd.read_csv("com_cheese_api/study/data/cheese_data.csv")

# def dummy_matching():
#     cheese_matching = cheese.matching.str.split('\s*,\s*', expand = True).stack().str.get_dummies().sum(level=0)
#     print(cheese_matching)
#     cheese_matching.to_csv(os.path.join('com_cheese_api/study/data', 'cheese_matching.csv'), index=True, encoding='utf-8-sig')

# dummy_matching()



def matching_list():
    matching_spacing = cheese.matching.str.split(',')
    # print(matching_spacing)
    cheese_matching = matching_spacing.apply(lambda x: pd.Series(x))

    # 컬럼을 행으로 변환 + matching 낱개만 가져오기
    cheese_matching.stack().reset_index(level = 1, drop = True).to_frame('matching_single')

    matching_list = cheese.merge(cheese_matching, left_index = True, right_index = True, how = 'left')
    matching_list.to_csv(os.path.join('com_cheese_api/study/data', 'cheese_food.csv'), index=True, encoding='utf-8-sig')
    
    print(matching_list)



matching_list()