import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('mental.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('psycho.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ["currently_employed "," computer "," hospitalized_before_mental_illness ","days_hospitalized","legally_disabled","regular_access_internet","live_with_parents","a_gap_resume","Total_length_any_gaps_resume_months","unemployed","read_outside_work_school","hospitalized_mental_illness","Lack_of_concentration","Anxiety","Depression","Obsessive_thinking","Mood_swings"," Panic_attacks"," Compulsive_behavior "," Tiredness"," Age","Education_Completed Masters","Education_Completed Phd","Education_Completed Undergraduate"," Education_MADHYAMIK OR HIGHERSECONDARY","Education_Some Phd","Education_Some Undergraduat","Education_Some highschool","Education_Some Masters","Gender_Female","Gender_Male"]
	
    
    df= pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)
        
    if output == 1:
        res_val = "** Mental Disorder,Please  Consult to the Psychiatrist**"
    else:
        res_val = " **No Mental Disorder,Enjoy Your Life**"
        

    return render_template('psycho.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == "__main__":
    app.run()
