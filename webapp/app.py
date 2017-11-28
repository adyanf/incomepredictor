from flask import Flask, request, render_template, jsonify
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

app = Flask(__name__)

def relationship(x):
    if (x == 'Husband' or x == 'Wife'):
        return 'Husband-Wife'
    else:
        return 'AHW'
    
def tipe_jam(x):
    if (x >= 48 and x <= 60):
        return 1.125
    return 0

@app.route('/')
def home():
	return jsonify('this just only api')

@app.route('/getincome',methods=['POST','GET'])
def get_income():
    if request.method=='POST':
        result = request.form

        y = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
                
        df = []
        for x in y:
            if (x in ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']):
                df.append((x, [int(result[x])]))
            else:  
                df.append((x, [result[x]]))
        dfs = pd.DataFrame.from_items(df)

        # print (df)
        # print (dfs)

        enc = LabelEncoder()
        categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

        dfs['relationship'] = [relationship(x) for x in dfs['relationship']]
        dfs['hours-per-week'] = [tipe_jam(x) for x in dfs['hours-per-week']]

        for x in categorical:
            enc.classes_ = np.load('models/' + x + '.npy')
            dfs[x] = enc.transform(dfs[x])


        drop_fitur = ['fnlwgt', 'education', 'race']
        dfs = dfs.drop(drop_fitur, axis=1)

		#Prepare the feature vector for prediction
        loaded_model = joblib.load('models/best.model')
        prediction = loaded_model.predict(dfs)

        return jsonify(prediction[0])

    
if __name__ == '__main__':
	app.run()