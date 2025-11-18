from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
app.secret_key = 'depression_app_2025'

# 10 Questions
questions = [
    {"id": "sadness", "text": "How often do you feel sad or down?", "options": ["Never (0)", "Rarely (1)", "Sometimes (2)", "Often (3)", "Always (4)"]},
    {"id": "sleep_disturbance", "text": "How often do you have trouble sleeping?", "options": ["Never (0)", "Rarely (1)", "Sometimes (2)", "Often (3)", "Always (4)"]},
    {"id": "body_pain", "text": "How often do you feel body pain or heaviness?", "options": ["Never (0)", "Rarely (1)", "Sometimes (2)", "Often (3)", "Always (4)"]},
    {"id": "loss_of_interest", "text": "How often do you lose interest in things you used to enjoy?", "options": ["Never (0)", "Rarely (1)", "Sometimes (2)", "Often (3)", "Always (4)"]},
    {"id": "fatigue", "text": "How often do you feel tired or have low energy?", "options": ["Never (0)", "Rarely (1)", "Sometimes (2)", "Often (3)", "Always (4)"]},
    {"id": "guilt", "text": "How often do you feel worthless or guilty?", "options": ["Never (0)", "Rarely (1)", "Sometimes (2)", "Often (3)", "Always (4)"]},
    {"id": "concentration", "text": "How often do you have trouble concentrating?", "options": ["Never (0)", "Rarely (1)", "Sometimes (2)", "Often (3)", "Always (4)"]},
    {"id": "income_level", "text": "What is your income level?", "options": ["Low (0)", "Medium (1)", "High (2)"]},
    {"id": "employment_status", "text": "What is your employment status?", "options": ["Unemployed (0)", "Employed (1)", "Student (2)"]},
    {"id": "hopelessness", "text": "How often do you feel hopeless about the future?", "options": ["Never (0)", "Rarely (1)", "Sometimes (2)", "Often (3)", "Always (4)"]},
]

# Load & Train Model
csv_path = 'nigerian_depression_data.csv'
if os.path.exists(csv_path):
    data = pd.read_csv(csv_path)
    le = LabelEncoder()
    if 'income_level' in data.columns:
        data['income_level'] = le.fit_transform(data['income_level'])
    if 'employment_status' in data.columns:
        data['employment_status'] = le.fit_transform(data['employment_status'])
    
    X = data.drop('severity', axis=1)
    y = data['severity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"Model trained! MSE: {mean_squared_error(y_test, model.predict(X_test)):.2f}")
else:
    model = None
    print("No CSV found. Using average score.")

@app.route('/', methods=['GET', 'POST'])
def welcome():
    if request.method == 'POST':
        if request.form.get('consent'):
            session['answers'] = {}
            session['current_question'] = 0
            return redirect(url_for('questionnaire'))
    return render_template('welcome.html')

@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    if 'current_question' not in session:
        return redirect(url_for('welcome'))
    
    q_index = session['current_question']
    if request.method == 'POST':
        answer = request.form.get('answer')
        if answer:
            session['answers'][questions[q_index]['id']] = int(answer.split('(')[1].split(')')[0])
            session['current_question'] += 1
            if session['current_question'] >= len(questions):
                return redirect(url_for('predict'))
    
    if q_index >= len(questions):
        return redirect(url_for('predict'))
    
    question = questions[q_index]
    progress = (q_index / len(questions)) * 100
    return render_template('question.html', q_num=q_index+1, question=question['text'], 
                          options=question['options'], progress=progress)

@app.route('/predict')
def predict():
    if 'answers' not in session:
        return redirect(url_for('welcome'))
    
    inputs = [session['answers'].get(q['id'], 0) for q in questions]
    
    if model:
        score = model.predict([inputs])[0]
    else:
        score = sum(inputs) / len(questions) * 2.5  # Scale to 0-10
    
    score = max(0, min(10, round(score, 1)))
    
    if score <= 3:
        alert, msg = "success", "Low risk – Keep up healthy habits!"
    elif score <= 6:
        alert, msg = "warning", "Moderate risk – Consider talking to someone."
    else:
        alert, msg = "danger", "High risk – Please see a doctor or counselor."
    
    session.clear()
    return render_template('result.html', score=score, alert=alert, msg=msg)



app.run(debug=True)
