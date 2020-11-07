from flask import Flask
from flask_restful import Api
from com_cheese_api.ext.db import url, db
# from com_cheese_api.ext.routes import initialize_routes
# from com_cheese_api.resources.user import UserDao
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

CORS(app, resources={r'/api/*': {"origins": "*"}})

app.config['SQLALCHEMY_DATABASE_URI'] = url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(app)

db.init_app(app)
api = Api(app)
with app.app_context():
    db.create_all()

print('SUCCESS')
# initialize_routes(api)




# from flask import Flask
# from flask_restful import Api
# from flask_cors import CORS
# from com_cheese_api.ext.db import url, db
# from flask_sqlalchemy import SQLAlchemy

# app = Flask(__name__)

# config = {
#     'user': 'bitai',
#     'password': '456123',
#     'host': '127.0.0.1',
#     'port': '3306',
#     'database': 'com_cheese_api'
# }

# charset = {'utf8':'utf8'}

# url = f"mysql+mysqlconnector://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}?charset=utf8"



# app.config['SQLALCHEMY_DATABASE_URI'] = url
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(app)

# db.init_app(app)
# with app.app_context():
#     db.create_all()

# print('SUCCESS')