import os

#print("current working directory: " + os.getcwd())
#print("absolute path : " + os.path.abspath("\data"))
#print("find directory name: " + os.path.dirname(__file__))
print("new create directory path: " + os.path.join(os.path.abspath(os.path.dirname(__file__))+'/data'))