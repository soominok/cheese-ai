#-*-coding:utf-8-*-

import pandas as pd
import numpy as np

def change_list() :
    top_item = (['비타민', '김치', '삼다수', '풀무원', '생수', '제주', '홍삼', '유산균', '캡슐', '훈제', '증정', '사과', '쇼핑', '큐브', '락토핏', '고려', '은단', '야채', '밸런스', '라이트', '홍진경', '선물', '간편', '플러스', '스테이크', '도시락', '롯데', '곤약', '멀티', '콜라겐', '라이프','견과', '두유', '오메가', '네슬레', '아셉틱', '퓨어', '소시지', '청양고추', '칼슘', '젤리', '시래기', '솥밥', '표고버섯', '프리미엄', '건강', '포도', '우먼', '스틱', '종근당', '포장', '석류', '주스', '하루', '실속', '불고기', '만두', '백화점', '키즈', '자연', '볶음', '모음', '미니', '낱봉', '호두', '라이스', '마그네슘', '혼합', '루테', '영양', '볶음밥', '리지', '워터', '버섯', '샘물', '유기농', '아몬드', '계란', '간장','수제', '안국', '미네랄', '포맨', '노니', '정환', '에너지', '코어', '매콤', '실버', '굿데이', '견과류', '안심', '프로','밀크', '꼬치','테일러','식이섬유', '땅콩','가야','베이스'])
    #print(type(top_item))
    #print(top_item)

    item_data = pd.read_csv("com_cheese_api/resources/data/users.csv")
    item_df = item_data.loc[:,['cheese_name']]
    item_lists = np.array(item_df['cheese_name'].tolist())
    #print(type(item_lists))

    cheese_data = pd.read_csv("com_cheese_api/resources/data/cheese_data.csv")
    cheese_df = cheese_data.loc[:,['name']]
    cheese_lists = np.array(cheese_df['name']).tolist()
    print(cheese_lists)

    for item_list in item_lists:
        for item_num in range(len(item_lists)):
            if item_list in top_item:
                change_value_dict = {item_list[item_num]: cheese_lists[item_num]}
                change_item_data = item_data.replace({'cheese_name': change_value_dict})
                print(change_item_data)
            else:
                break
    print(change_value_dict)       
    return change_item_data

    
    
# for item_num in range(len(item_lists)):
#     item_list = item_lists[item_num]
#     if item_list in top_item:
#         for 
#        change_value_dict = {}


# if item_lists[0] in top_item:
#     print("list in")
# else:
#     print("list not in")
#     print(item_lists[0])
#     print(top_item)
