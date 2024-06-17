import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import mpld3

app = Flask(_name_)

model = pickle.load(open(r"C:\Users\nebul\Downloads\model.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('front.html')

@app.route('/survey', methods=['GET', 'POST'])
def survey():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    features = {
            'anxiety_level': int(request.form['anxiety_level']),
            'self_esteem': int(request.form['self_esteem']),
            'depression': int(request.form['depression']),
            'blood_pressure': int(request.form['blood_pressure']),
            'sleep_quality': int(request.form['sleep_quality']),
            'safety': int(request.form['safety']),
            'basic_needs': int(request.form['basic_needs']),
            'teacher_student_relationship': int(request.form['teacher_student_relationship']),
            'future_career_concerns': int(request.form['future_career_concerns']),
            'social_support': int(request.form['social_support']),
            'extracurricular_activities': int(request.form['extracurricular_activities']),
            'bullying': int(request.form['bullying']),
        }
    
    input_data = pd.DataFrame([features])
    prediction = model.predict(input_data)

    recommendation_anxiety = anxiety_level(features['anxiety_level'])
    recommendation_self_esteem = self_esteem(features['self_esteem'])
    recommendation_depression = depression(features['depression'])
    recommendation_blood_pressure = blood_pressure(features['blood_pressure'])
    recommendation_sleep_quality = sleep_quality(features['sleep_quality'])
    recommendation_safety = safety(features['safety'])
    recommendation_basic_needs = basic_needs(features['basic_needs'])
    recommendation_teacher_student_relationship = teacher_student_relationship(features['teacher_student_relationship'])
    recommendation_future_career_concerns = future_career_concerns(features['future_career_concerns'])
    recommendation_social_support = social_support(features['social_support'])
    recommendation_extracurricular_activities = extracurricular_activities(features['extracurricular_activities'])
    recommendation_bullying = bullying(features['bullying'])

    features_list = list(input_data.columns)
    importance = model.feature_importances_

    fig, ax = plt.subplots()
    bars = ax.bar(features_list, importance, color='skyblue')
    ax.set_title('Features Contribution in your Stress')
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')

    chart_html = mpld3.fig_to_html(fig)
    def call(n):
        if(prediction[0]==0):
            return "Stress is Low"
        elif(prediction[0]==1):
            return "Stress is Moderate"
        else:
            return "Stress is High"
    return render_template('result.html',
                           prediction_text= call(prediction[0]),
                           chart_html=chart_html,
                           recommendation_anxiety=recommendation_anxiety,
                           recommendation_self_esteem=recommendation_self_esteem,
                           recommendation_depression=recommendation_depression,
                           recommendation_blood_pressure=recommendation_blood_pressure,
                           recommendation_sleep_quality=recommendation_sleep_quality,
                           recommendation_safety=recommendation_safety,
                           recommendation_basic_needs=recommendation_basic_needs,
                           recommendation_teacher_student_relationship=recommendation_teacher_student_relationship,
                           recommendation_future_career_concerns=recommendation_future_career_concerns,
                           recommendation_social_support=recommendation_social_support,
                           recommendation_extracurricular_activities=recommendation_extracurricular_activities,
                           recommendation_bullying=recommendation_bullying)

def anxiety_level(feature_value):
    if feature_value <=0:
        return "Your anxiety level is low. Keep practicing stress-reducing activities and maintaining a positive mindset."
    elif 1 <= feature_value <= 3:
        return "Consider incorporating relaxation techniques into your routine to manage anxiety."
    else:
        return "Your anxiety level is high. It's important to seek support from friends, family, or professionals."

def self_esteem(feature_value):
    if  0<=feature_value < 1:
        return "Your self-esteem is low. Focus on your strengths and achievements, and consider seeking support from a mentor or counselor."
    elif 2 <= feature_value <= 3:
        return "Work on building your self-esteem by setting small goals and celebrating achievements."
    else:
        return "Your self-esteem is high. Keep nurturing a positive self-image and continue with confidence."

def depression(feature_value):
    if feature_value <=0:
        return "Your depression level is low. Engage in activities that bring you joy and consider maintaining a support system."
    elif 1 <= feature_value <= 3:
        return "Consider seeking professional help or talking to someone you trust about your feelings."
    else:
        return "Your depression level is high. It's crucial to reach out to a mental health professional for guidance and support."

def blood_pressure(feature_value):
    if feature_value <= 1:
        return "Your blood pressure is in a healthy range. Maintain a balanced diet and regular exercise for overall well-being."
    elif feature_value <= 2:
        return "Monitor your blood pressure regularly and consider lifestyle changes such as reducing salt intake and increasing physical activity."
    else:
        return "Your blood pressure is elevated. It's important to consult with a healthcare professional for personalized advice and monitoring."

def sleep_quality(feature_value):
    if 0<=feature_value <= 1:
        return "Your sleep quality may be affecting your stress level. Consult with a healthcare professional if sleep issues persist."
    elif 2 <= feature_value <= 3:
        return "If you're experiencing sleep disturbances, consider improving your sleep environment and practicing relaxation techniques before bedtime."
    else:
        return "Your sleep quality is good. Maintain a consistent sleep schedule and create a relaxing bedtime routine for optimal rest."

def safety(feature_value):
    if 0<=feature_value <= 1:
        return "If you're facing significant safety issues, seek immediate help from authorities and support services."
    elif 2 <= feature_value <= 3:
        return "If you have safety concerns, consider discussing them with a trusted adult or authority figure. Your well-being is important."
    else:
        return "You feel safe in your environment. Continue prioritizing your well-being and report any safety concerns to appropriate authorities."

def basic_needs(feature_value):    
    if feature_value <=0:
        return "Ensure your basic needs are met, including food, shelter, and clothing. Seek assistance from local resources if needed."
    elif 1 <= feature_value <= 2:
        return "If you're struggling to meet basic needs, reach out to community resources, social services, or local charities for support."
    elif 3<= feature_value <= 4:
        return "continue_normal."
    else:
        return "All your basic needs are met so continue normal."

def teacher_student_relationship(feature_value):
    if 0<=feature_value <= 1:
        return "If you're experiencing significant challenges in your teacher-student relationships, consider involving school counselors or administrators for assistance."
    elif 2<= feature_value <= 3:
        return "If you have challenges in your teacher-student relationships, consider discussing them with your teachers to find solutions together."
    else:
        return "Maintain open communication with your teachers. If you have concerns, consider discussing them with your teachers for better support."
           
def future_career_concerns(feature_value):
    if feature_value <=0:
        return "since you have no career concern that might not affect your mental stress."
    elif 1 <= feature_value <= 2:
        return "If you have concerns about your future career, explore career guidance resources and consider discussing your goals with a career counselor."
    elif 3<= feature_value <= 4:
        return "If you're uncertain about your future career, seek guidance from career counselors, mentors, or professionals in your field of interest."
    else:
        return "If you're experiencing significant stress about your future career, consider seeking support from career counselors, mentors, or mental health professionals."

def social_support(feature_value):
    if feature_value <= 0:
        return "If you lack a strong social support network, consider joining clubs, organizations, or seeking professional help to build meaningful connections."
    elif 1 <= feature_value <= 2:
        return "Maintain and nurture your social connections. If you're feeling isolated, reach out to friends and family for support."
    else:
        return "Strengthen your social support network by connecting with friends, family, and supportive individuals. Social connections can positively impact stress levels."

def extracurricular_activities(feature_value):
    if feature_value <= 0:
        return "Surely Engage yourself in extracurricular activities that bring you joy and relaxation. Hobbies and interests can positively impact stress levels."
    elif 1 <= feature_value <= 2:
        return "Always try to  participate in extracurricular activities that interest you. They can provide a positive outlet for stress relief."
    elif 3 <= feature_value <= 4:
        return "Continue participating in extracurricular activities that interest you. They can provide a positive outlet for stress relief."
    else:
        return "Continue participating in extracurricular activities that interest you. They can provide a positive outlet for stress relief."

def bullying(feature_value):
    if feature_value <=0:
        return "You're not experiencing significant bullying. If you encounter any challenges, consider discussing them with a trusted adult."
    elif 1 <= feature_value <= 2:
        return "If you're facing slight bullying, it's important to speak up and seek support from teachers, parents, or school authorities."
    elif 3<= feature_value <= 4:
        return "If you're facing significant  bullying, it's important to speak up and seek support from teachers, parents, or school authorities."
    else:
        return "If you're experiencing severe bullying, don't hesitate to involve authorities and seek professional help."
    
if _name_ == "_main_":
    app.run()