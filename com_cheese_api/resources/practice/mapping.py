import pandas as pd
import numpy as np
users = pd.read_csv('com_cheese_api/resources/data/users.csv')

# age_mapping = {'10': 1, '20': 1, '30': 2, '40': 3, '50': 4, '60': 5, '70': 5, '80': 5}
# for dataset in users:
#         dataset['age_group'] = dataset['user_age'].map(age_mapping)
# this.train = this.train
# this.test = this.test
# return this

def user_age_norminal():
        bins = [19, 29, 39, 49, 59, np.inf]
        labels = ['Youth', 'Adult30', 'Adult40', 'Adult50', 'Senior']
        users['age_group'] = pd.cut(users['user_age'], bins, right = True, labels = labels)
        print(users)
        age_mapping = {
                'Youth': 1,
                'Adult30': 2 ,
                'Adult40': 3 ,
                'Adult50': 4,
                'Senior': 5
        }
        users['age_group'] = users['age_group'].map(age_mapping)
        print(users)
        users.to_csv("com_cheese_api/resources/data/11.csv")


# def user_age_norminal():
#         age_mapping = {
#                 10: 1,
#                 20: 1,
#                 30: 2 ,
#                 40: 3 ,
#                 50: 4,
#                 'Senior': 5
#         }
#         age_func = lambda x: age_mapping.get(x, x)

user_age_norminal()
