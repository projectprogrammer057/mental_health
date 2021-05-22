import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('life.pickle', 'rb'))

@app.route('/')
def home():
    return render_template('psycho.html')


@app.route('/predict',methods=["POST"])
def predict():
    if request.method == "POST":
        employed = request.form['currently_employed']
        disabled = request.form['legally_disabled']
        internet = request.form['regular_access_internet']
        parent = request.form['live_with_parents']
        concentration = request.form['Lack_of_concentration']
        anxiety = request.form['Anxiety']
        depression = request.form['Depression']
        opcesive = request.form['Obsessive_thinking']
        mood = request.form['Mood_swings']
        panic = request.form['Panic_attacks']
        compulsive = request.form['Compulsive_behavior']
        tiredness = request.form['Tiredness']
        age = request.form['Age']
        female = request.form['Female']
        male = request.form['Male']
       
        features_value = [np.array([employed,disabled,internet,parent,concentration,anxiety,depression,opcesive,mood,panic,compulsive,tiredness,age,female,male])]
    
        features_name = ["currently_employed ","legally_disabled","regular_access_internet","live_with_parents","Lack_of_concentration","Anxiety","Depression","Obsessive_thinking","Mood_swings","Panic_attacks","Compulsive_behavior ","Tiredness","Age","Female","Male"]
	
    
        df= pd.DataFrame(features_value, columns=features_name)
        output = model.predict(df)
        
        if output == 1:
            res_val = "** Mental Disorder,Please  Consult to the Psychiatrist**"
        else:
            res_val = " **No Mental Disorder,Enjoy Your Life**"
        

    return render_template('psycho.html', prediction_text='Patient has {}'.format(res_val))




if __name__ == "__main__":
    app.run(debug=True)
