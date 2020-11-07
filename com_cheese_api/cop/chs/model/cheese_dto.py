from com_cheese_api.ext.db import url, db, openSession, engine
from com_cheese_api.util.file import FileReader
from flask import request
from sqlalchemy import func
from sqlalchemy import and_, or_
from flask import Response, jsonify
from flask_restful import Resource, reqparse
from sklearn.ensemble import RandomForestClassifier # rforest
from sklearn.tree import DecisionTreeClassifier # dtree
from sklearn.ensemble import RandomForestClassifier # rforest
from sklearn.naive_bayes import GaussianNB # nb
from sklearn.neighbors import KNeighborsClassifier # knn
from sklearn.svm import SVC # svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold  # k value is understood as count
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import json
import os
import sys
from typing import List
from pathlib import Path

class CheeseDto(db.Model):
    __tablename__='cheeses'
    __table_args__={'mysql_collate':'utf8_general_ci'}

    ranking : int = db.Column(db.Integer, primary_key=True, index=True)
    brand : int = db.Column(db.Integer)
    category: int = db.Column(db.Integer)
    types : int = db.Column(db.Integer)
    texture : str = db.Column(db.String(30))
    img : str = db.Column(db.String(255))


    #dairy = db.relationship('DiaryDto', lazy='dynamic')
    # orders = db.relationship('OrderDto', back_populates='cheese', lazy='dynamic')
    # prices = db.relationship('PriceDto', back_populates='cheese', lazy='dynamic')

    def __init__(self, ranking, brand, category, types, texture, img) : 
        self.ranking = ranking
        self.brand = brand
        self.category = category
        self.types = types
        self.texture = texture
        self.img = img

    def __repr__(self):
        return f'Cheese(ranking={self.ranking}, brand={self.brand}, category={self.category}, \
                    types={self.types}, texture={self.texture}, img={self.img})'

    def __str__(self):
        return f'Cheese(ranking={self.ranking}, brand={self.brand}, category={self.category}, \
                    types={self.types}, texture={self.texture}, img={self.img})'


    @property
    def json(self):
        return {'ranking':self.ranking, 'brand':self.brand, 'category':self.category, \
                    'types':self.types, 'texture':self.types, 'img':self.img}

class CheeseVo():
    ranking : 0
    brand : ''
    category: 0
    types : 0
    texture : ''
    img : ''

