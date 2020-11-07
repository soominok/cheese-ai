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


# DB로 데이터 전송하는 부분
class ReviewDto(db.Model):

    __tableName__="reviews"
    __table_args__={'mysql_collate':'utf8_general_ci'}

    review_no: int = db.Column(db.Integer, primary_key=True, index=True)
    category: int = db.Column(db.Integer)
    brand: str = db.Column(db.String(255))
    cheese_name: str = db.Column(db.String(255))
    review_title: str = db.Column(db.String(100))
    review_detail: str = db.Column(db.String(500))
    review_views: int = db.Column(db.Integer)

#     user_id = db.Column(db.String(10), db.ForeignKey(ReviewDto.user_id))
#     user = db.relationship('ReviewDto', back_populates='reviews')
#     item_id = db.Column(db.Integer, db.ForeignKey(ItemDto.item_id))
#     item = db.relationship('ItemDto', back_populates='reviews')

    def __init__(self,  review_no, category, brand, cheese_name, review_title, review_detail, review_views):
        self.review_no = review_no,
        self.category = category,
        self.brand = brand,
        self.cheese_name = cheese_name,
        self.review_title = title,
        self.review_detail = review_detail,
        self.review_views = review_views


    def __repr__(self):
        return f'review_no={self.review_no}, category={self.category}, brand={self.brand},\
            cheese_name={self.cheese_name}, review_title={self.review_title}, review_detail={self.review_detail}, \
                review_views={self.review_views}'

    def json(self):
        return {
            'review_no': self.review_no,
            'category': self.category,
            'brand': self.brand,
            'cheese_name': self.cheese_name,
            'review_title': self.review_title,
            'review_detail': self.review_detail,
            'review_views': self.review_views
        }


class ReviewVo():
    review_no: int = 0
    category: str = 0
    brand: int = ''
    cheese_name: str = ''
    review_title: str = ''
    review_detail: str = ''
    review_views: int = 0