import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns


users = pd.read_csv("com_cheese_api/resources/data/users.csv")
# print(users)

def itemList_count():
    # font_location = '/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf'
    # font_name = fm.FontProperties(fname = font_location).get_name()
    # plt.rc('font', family=font_name)
    # plt.rc('font', family='NanumBarunGothicBold')

    item_count_plot = sns.barplot (x = users['sub1_category'].value_counts().index, y = users['sub1_category'].value_counts().values, data = users)
    # item_count_plot.set_title('아이템 카테고리별 빈도 수', fontproperties=font_name)
    # item_count_plot.set_xlabel('아이템 카테고리 종류', fontproperties=font_name)
    # item_count_plot.set_ylabel('빈도 수', fontproperties=font_name)
    # plt.xticks(rotation = 90, fontproperties=font_name)
    item_count_plot.set_title('아이템 카테고리별 빈도 수')
    item_count_plot.set_xlabel('아이템 카테고리 종류')
    item_count_plot.set_ylabel('빈도 수')
    plt.xticks(rotation = 90)
    plt.show()
    

#list_count()
itemList_count()