# from statistics import linear_regression
# from flask import Flask, render_template, request
# import numpy as np
# from sklearn.linear_model import LinearRegression
# import joblib

# app = Flask(__name__)

# # Placeholder for loading dataset and training the model
# def load_dataset():
#     # Load your dataset using pandas or any other library
#     # Example:
#     # data = pd.read_csv('your_dataset.csv')
#     # return data
#     pass

# def train_model(data):
#     # Prepare your features (X) and target (y) from the loaded dataset
#     # Example:
#     # features = ['Age', 'Gender', 'Profession', 'Income', 'Savings', 'Assets']
#     # target = 'Expense'
#     # X = data[features]
#     # y = data[target]

#     # Create an instance of the LinearRegression model
#     model = LinearRegression()

#     # Train the model
#     # model.fit(X, y)
#     return model

# # Load your dataset and train the model
# # data = load_dataset()
# # model = train_model(data)

# linear_regression_model = joblib.load('linear_regression_model.pkl')
# joblib.dump(linear_regression, 'linear_regression_model.pkl')

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     age = int(request.form['age'])
#     gender = int(request.form['gender'])
#     profession = int(request.form['profession'])
#     income = float(request.form['income'])
#     savings = float(request.form['savings'])
#     assets = float(request.form['assets'])

#     # Prepare the input data for prediction
#     X_new = np.array([age, gender, profession, income, savings, assets]).reshape(1, -1)

#     # Make the prediction
#     y_pred = linear_regression_model.predict(X_new)

#     # Assuming y_pred is an array of predicted values for different categories
    
#     return render_template('result.html', prediction=y_pred)

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, request
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import pickle

app = Flask(__name__)

# Placeholder for loading dataset and training the model
def load_dataset():
    # Load your dataset using pandas or any other library
    # Example:
    # data = pd.read_csv('your_dataset.csv')
    # return data
    pass

def train_model(data):
    # Prepare your features (X) and target (y) from the loaded dataset
    # Example:
    # features = ['Age', 'Gender', 'Profession', 'Income', 'Savings', 'Assets']
    # target = 'Expense'
    # X = data[features]
    # y = data[target]

    # Create an instance of the LinearRegression model
    model = LinearRegression()

    # Train the model
    # model.fit(X, y)
    return model

# Load your dataset and train the model
# data = load_dataset()
# model = train_model(data)

# Assuming you have already trained the model and saved it as 'linear_regression_model.pkl'
linear_regression_model = joblib.load('linear_regression_model.pkl')

# Define the list of professions
professions = [
    ('', '--Select--'),
    (0, 'Accountant'),
    (1, 'Architect'),
    (3, 'Business Owner'),
    (2, 'Banker'),
    (4, 'Consultant'),
    (5, 'Content Writer'),
    (7, 'Doctor'),
    (6, 'Data Analyst'),
    (8, 'Engineer'),
    (9, 'Entrepreneur'),
    (10, 'Financial Analyst'),
    (11, 'Graphic Designer'),
    (12, 'HR Manager'),
    (13, 'IT Consultant'),
    (15, 'Lawyer'),
    (16, 'Marketing Manager'),
    (17, 'Marketing Specialist'),
    (18, 'Nurse'),
    (21, 'Project Manager'),
    (19, 'Pharmacist'),
    (20, 'Professor'),
    (22, 'Sales Representative'),
    (23, 'Software Developer'),
    (25, 'Student'),
    (24, 'Software Engineer'),
    (26, 'Teacher'),
]

# Generate the HTML code
options = ""
for profession_id, profession_name in professions:
    options += f'<option value="{profession_id}">{profession_name}</option>\n'

# Save the HTML code to a file (optional)
with open('professions_options.html', 'w') as file:
    file.write(options)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/index')
def home():
    return render_template('index.html', options=options)

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    gender = int(request.form['gender'])
    profession = int(request.form['profession'])
    income = float(request.form['income'])
    savings = float(request.form['savings'])
    assets = float(request.form['assets'])

    # Prepare the input data for prediction
    X_new = np.array([age, gender, profession, income, savings, assets]).reshape(1, -1)

    # Make the prediction
    y_pred = linear_regression_model.predict(X_new)
   

    # Assuming y_pred is an array of predicted values for different categories
    # Modify this according to your actual model output

    return render_template('result.html', total_expense=y_pred[0, 0])

if __name__ == '__main__':
    app.run(debug=False)

with open('linear_regression_model.pkl', 'rb') as file:
    linear_regression_model = pickle.load(file)