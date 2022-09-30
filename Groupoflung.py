import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import joblib
import pickle 
from pickle import dump
from pickle import load




def main():
    st.title('Stroke Prediction by Team of Lung')
    
        #Caching the model for faster loading
        # @st.cache
        # def predict(fixed_acidity, volatile_acidity,  citric_acid, residual_Sugar, Free_sulfur_dioxide, Total_sulfur_dioxide, Merry or not, age ):
    col1, col2 = st.columns(2)
    with col1:
            gender = st.selectbox('Gender:', ["Female", "Male"])
            if gender == "Female":
                gender = 0
            elif gender == 'Male':
                gender = 1
        
            hypertension = st.selectbox("Hypertesnion", ["No", "Yes"])
            if hypertension == "No":
                hypertension = 0
            else:
                hypertension = 1

            ever_married = st.selectbox("Married", ["No", 'Yes'])
            if ever_married == "No":
                ever_married = 0
            else:
                ever_married = 1

            Residence_type = st.selectbox("Resident Area", ['Rural', 'Urban'])
            if Residence_type == "Rural":
                Residence_type = 0
            else:
                Residence_type = 1

            bmi = st.number_input("Body Mass Index", min_value=8.0, max_value=100.0)
    with col2:
            age = st.number_input('Age:', min_value=1, max_value=110)

            heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
            if heart_disease == "No":
                heart_disease = 0
            else:
                heart_disease = 1

            work_type = st.selectbox("Work Type", ['Private', 'Self-employed', 'Government Job', 'Child', 'Never Worked'])
            if work_type == 'Child':
                work_type = 0
            elif work_type == 'Never Worked':
                work_type = 1
            elif work_type == 'Private':
                work_type = 2
            elif work_type == 'Self-employed':
                work_type = 3
            else:
                work_type = 4

            avg_glucose_level = st.number_input("Average Glucose Level", min_value=30.0, max_value=300.0)

            smoking_status = st.selectbox("Smoking Status", ['Formerly smoked', 'Never smoked', 'Smokes', 'Unknown'])
            if smoking_status == 'Unknown':
                smoking_status = 0
            elif smoking_status == 'Formerly smoked':
                smoking_status = 1
            elif smoking_status == 'Never smoked':
                smoking_status = 2
            else:
                smoking_status = 3


    input_dict = {'gender':gender, 'age':age, 'hypertension':hypertension, 'heart_disease':heart_disease,
        'ever_married':ever_married, "work_type":work_type, 'Residence_type':Residence_type, "avg_glucose_level":avg_glucose_level, 
        'bmi':bmi, 'smoking_status':smoking_status}
    input_df = pd.DataFrame(input_dict, index=[0])
        ## predict button
    button = st.button('Predict')

    if button:
        st.write(input_df)
        #pickle_transformer = open('transformer_entrenado.pkl', 'rb') 
        carga_transformer = pickle.load(open('transformer_entrenado.pkl', 'rb'))
        #pickle_in = open('XGBClassifier.pkl', 'rb') 
        carga_modelo = pickle.load(open('XGBClassifier.pkl', 'rb'))
        transformer = carga_transformer
        modelo = carga_modelo
        df = transformer.transform(input_df)
        st.write(df)
        predict=model.predict(df)





if __name__=='__main__': 
    main() 