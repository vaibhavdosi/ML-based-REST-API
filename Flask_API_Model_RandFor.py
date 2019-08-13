import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators = 10000, random_state = 0)
regressor.fit(X,y)

#model save and load using pickle or joblib
#To save 
import pickle 
pickle.dump(regressor,open('objectFile_','wb'))




from flask import Flask,request,jsonify
from flask_cors import CORS, cross_origin
from sklearn.externals import joblib
import pandas as pd

app=Flask(__name__)
CORS(app)

headers=['Level']
f=open('objectFile_','rb')

#To load model
loaded_model=pickle.load(open('objectFile_','rb'))


@app.route('/predict',methods=['POST'])
def predict():
    payload=request.json['data']
    inpu=pd.DataFrame([payload],columns=headers,dtype=int,index=['input'])
    pre=loaded_model.predict(inpu)
    ret='{"prediction":'+str(pre) +'}'
    return ret

if __name__ == '__main__':
    app.run()

