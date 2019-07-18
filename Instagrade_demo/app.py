from flask import Flask,render_template,url_for,request

import pandas as pd
import numpy as np
from fastai.text import *
import math

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = message

		t1 = load_learner('/Users/RithwikSudharsan 1/Desktop/Instagrade/models/', file='t1model.pkl')
		t2 = load_learner('/Users/RithwikSudharsan 1/Desktop/Instagrade/models/', file='t2model.pkl')
		t3 = load_learner('/Users/RithwikSudharsan 1/Desktop/Instagrade/models/', file='t3model.pkl')
		t4 = load_learner('/Users/RithwikSudharsan 1/Desktop/Instagrade/models/', file='t4model.pkl')
		t5 = load_learner('/Users/RithwikSudharsan 1/Desktop/Instagrade/models/', file='t5model.pkl')
		t6 = load_learner('/Users/RithwikSudharsan 1/Desktop/Instagrade/models/', file='t6model.pkl')

		def temp(a):
			return((math.log(2+math.sqrt(3)))/10)

		def sigmoid(x):
			return 1 / (1 + math.exp(-temp(x)))

		t1pred=str(round((t1.predict(data)[0].data[0]*100)/9,4))
		t2pred=str(round((t2.predict(data)[0].data[0]*100)/9,4))
		t3pred=str(round((t3.predict(data)[0].data[0]*100)/9,4))
		t4pred=str(round((t4.predict(data)[0].data[0]*100)/9,4))
		t5pred=str(round((t5.predict(data)[0].data[0]*100)/9,4))
		t6pred=str(round((t6.predict(data)[0].data[0]*100)/9,4))



		buffer="-"*146
		my_prediction = "Ideas and Content: {}% {} Organization: {}% {} Voice: {}% {} Word Choice: {}% {} Sentence Fluency: {}% {} Conventions: {}% {}"\
			.format(t1pred,buffer,t2pred,buffer,t3pred,buffer,t4pred,buffer,t5pred,buffer,t6pred,buffer)


	return render_template('result.html',prediction=my_prediction)


if __name__ == '__main__':
	app.run(debug=False)
