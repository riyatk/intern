from flask import Flask, render_template, request
import pickle
import numpy as np
import re

app=Flask(__name__)

# our ML model should be saved in the same folder as our main.py in this example:
# rb is the mode we open this pickle model - stands for "read / binary"
loaded_model = pickle.load(open("adaboost_classificationmodel.pkl","rb"))

@app.route('/')
def home():
    return render_template("home.html")


def ValuePredictor(np_arr):   
    result = loaded_model.predict(np_arr)
    return result[0]

def string_to_int(s):
    integer = re.sub('[^0-9]','', s)
    return int(integer)

def one_hot_input(s, length):
    index = string_to_int(s)
    arr = [0 for i in range(length)]
    arr[index] = 1
    return arr

def flatten_arr(arr):
    flat_list = []
    for item in arr:
        if isinstance(item, (list)):
            for it in item:
                flat_list.append(it)
        else:
            flat_list.append(item)
    return flat_list

# when the form press submit, it links it to the action /result which will be sent here:
@app.route('/result', methods = ['POST'])
def result():
    prediction=''
    if request.method == 'POST':
        # from the request form, convert it to a dictionary saved as this variable
        features = request.form.to_dict()

        ### converting inputs to their correct value types:
        features['age'] = string_to_int(features['age'])
        features['education_num'] = string_to_int(features['education_num'])
        features['capital_gain'] = string_to_int(features['capital_gain'])
        features['capital_loss'] = string_to_int(features['capital_loss'])
        features['hours_per_week'] = string_to_int(features['hours_per_week'])
        features['occupation'] = one_hot_input(features['occupation'], 15)
        features['relationship'] = one_hot_input(features['relationship'], 6)
        features['race'] = one_hot_input(features['race'], 5)

        print(features) # flag
        print(features.values()) # flag

        # get the values and turn it into a list
        features=list(features.values())

        # flatten the list:
        features = flatten_arr(features)

        # reshape the list into a np array: (with unknown amount of columns)
        features = np.array(features).reshape(1,-1)

        print("Before sending to model", features) # flag

        # sending to our prediction model (which will reshape it as a numpy array and then return our result)
        result = ValuePredictor(features)

        print("result from model", result) # flag

        if int(result)==0:
            prediction='Your salary is LESS than $50,000 USD per year.'
        else:
            prediction='Your salary is MORE than $50,000 USD per year.'
        
        print(prediction) # flag
        
        # passing the string of our prediction to our template
        return render_template("result.html",prediction=prediction) 
        

if __name__ == "__main__":
    app.run()


