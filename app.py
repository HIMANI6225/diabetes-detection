from flask import Flask, request, jsonify, render_template, redirect, flash, send_file
import pickle
import pandas as pd

app = Flask(__name__)  # Initialize the flask App

stacking = pickle.load(open('stacking.pkl','rb'))
lightgbm = pickle.load(open('lightgbm.pkl','rb'))
catboost = pickle.load(open('catboost.pkl','rb'))
ExtraTreeClassifier = pickle.load(open('ExtraTreeClassifier.pkl','rb'))

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/preview', methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset, encoding='unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html", df_view=df)

@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
       
        try:
            Pregnancies = float(request.form['Pregnancies'])
            Glucose = float(request.form['Glucose'])
            BloodPressure = float(request.form['BloodPressure'])
            SkinThickness = float(request.form['SkinThickness'])
            Insulin = float(request.form['Insulin'])
            bmi = float(request.form['bmi'])
            PedigreeFunction = float(request.form['PedigreeFunction'])
            Age = float(request.form['Age'])
            Model = request.form['Model']
        except ValueError:
            return "Invalid input. Please ensure all fields contain numeric values."

       
        input_variables = pd.DataFrame([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, bmi, PedigreeFunction, Age]],
                                       columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'],
                                       index=['input'])

        print(input_variables)

        
        if Model == 'StackingClassifier':
            prediction = stacking.predict(input_variables)
        elif Model == 'LGBMClassifier':
            prediction = lightgbm.predict(input_variables)
        elif Model == 'CatBoostClassifier':
            prediction = catboost.predict(input_variables)
        elif Model == 'ExtraTreeClassifier':
            prediction = ExtraTreeClassifier.predict(input_variables)
        else:
            return "Invalid model selected."

         
        outputs = prediction[0]
        results = "Diabetes Present" if outputs == 1 else "No Diabetes"

    return render_template('result.html', 
                           prediction_text=results, 
                           model=Model, 
                           Pregnancies=Pregnancies, 
                           Glucose=Glucose, 
                           BloodPressure=BloodPressure, 
                           SkinThickness=SkinThickness, 
                           Insulin=Insulin, 
                           bmi=bmi, 
                           PedigreeFunction=PedigreeFunction, 
                           Age=Age)


@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/performance')
def performance():
    return render_template('performance.html')

if __name__ == "__main__":
    app.run(debug=True)
