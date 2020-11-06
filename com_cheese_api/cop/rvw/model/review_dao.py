from com_cheese_api.cop.rvw.model.review_dfo import ReviewDf
from com_cheese_api.cop.rvw.model.review_dto import ReviewDto
import pandas as pd
from com_cheese_api.util.file import FileReader

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from com_cheese_api.ext.db import url, db, openSession, engine
from time import sleep
import time
import random
from urllib.request import urlopen
from bs4 import BeautifulSoup
from bs4 import BeautifulSoup
from bs4.element import NavigableString

from sklearn.model_selection import train_test_split

import os
from pathlib import Path


Session = openSession()
session = Session()

class ReviewDao(ReviewDto):
    @staticmethod
    def bulk():
        reviewDf = ReviewDf()
        df = reviewDf.new()
        print(df.head())
        session.bulk_insert_mappings(ReviewDto, df.to_dict(orient="records"))
        session.commit()
        session.close()

    @classmethod
    def find_all(cls):
        return cls.query.all()

    @classmethod
    def find_by_name(cls, name):
        return cls.query.filer_by(name == name).all()

    @classmethod
    def find_by_id(cls, id):
        return cls.query,filter(ReviewDto.rev_id == id).one()

    @staticmethod
    def save(review):
        Session = openSession()
        session = Session()
        session.add(review)
        session.commit()

    @staticmethod
    def update(review, review_no):
        Session = openSession()
        session = Session()
        session.query(ReviewDto).filter(ReviewDto.review_no == review.review_no)\
            .update({ReviewDto.review_title: review.review_title,
                        ReviewDto.review_detail: review.review_detail})
        session.commit()

    @classmethod
    def delete(cls, rev_id):
        Session = openSession()
        session = Session()
        cls.query(ReviewDto.review_no == review.review_no).delete()
        session.commit()



if __name__ == '__main__':
    ReviewDao.bulk()