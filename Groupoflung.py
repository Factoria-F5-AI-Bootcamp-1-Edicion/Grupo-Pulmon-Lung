import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import joblib
import pickle 
from pickle import dump
from pickle import load
import urllib
import urllib.request

def main():
    st.title('Stroke Prediction by Team of Lung')
    
        #Caching the model for faster loading
        # @st.cache
        # def predict(fixed_acidity, volatile_acidity,  citric_acid, residual_Sugar, Free_sulfur_dioxide, Total_sulfur_dioxide, Merry or not, age ):
    col1, col2 = st.columns(2)
    with col1:
            gender = st.selectbox('Gender:', ["Female", "Male"])
            if gender == "Female":
                gender = "Female"
            else: 
                gender = "Male"
        
            hypertension = st.selectbox("Hypertension", ["No", "Yes"])
            if hypertension == "No":
                hypertension = "0"
            else:
                hypertension = "1"

            ever_married = st.selectbox("Married", ["No", 'Yes'])
            if ever_married == "No":
                ever_married = "No"
            else:
                ever_married = "Yes"

            Residence_type = st.selectbox("Resident type", ['Rural', 'Urban'])
            if Residence_type == "Rural":
                Residence_type = "Rural"
            else:
                Residence_type = "Urban"

            bmi= st.number_input("Body Mass Index", min_value=8.0, max_value=100.0)
    with col2:
            age = st.number_input('Age:', min_value=0 , max_value=110)

            heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
            if heart_disease == "No":
                heart_disease = "0"
            else:
                heart_disease = "1"

            work_type = st.selectbox("Work Type", ['Private', 'Self-employed', 'Government Job', 'Child', 'Never Worked'])
            if work_type == 'children':
                work_type = "children"
            elif work_type == 'Private':
                work_type = "Private"
            elif work_type == 'Self-employed':
                work_type = "Self-employed"
            elif work_type == 'Govt_job':
                work_type = "Govt_job"
            

            avg_glucose_level = st.number_input("Average Glucose Level", min_value=30.0, max_value=300.0)

            smoking_status = st.selectbox("Smoking Status", ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])
            if smoking_status == 'Unknown':
                smoking_status = "Unknown"
            elif smoking_status == 'formerly smoked':
                smoking_status = "formerly smoked"
            elif smoking_status == 'never smoked':
                smoking_status = "never smoked"
            elif smoking_status == 'smokes' :
                smoking_status = "smokes"
   
            
         

    input_dict = {'gender':gender, 'age':age, 'hypertension':hypertension, 'heart_disease':heart_disease,
        'ever_married':ever_married, "work_type":work_type, 'Residence_type':Residence_type, "avg_glucose_level":avg_glucose_level, 
        'bmi':bmi, 'smoking_status':smoking_status}
    input_df = pd.DataFrame(input_dict, index=[0])
        ## predict button
    button = st.button('Predict')

    if button:
        st.write(input_df)
        #pickle_transformer = open('transformer_entrenado.pkl', 'rb') 
        carga_transformer = pickle.load(open('transformer_final.pkl', 'rb'))
        #pickle_in = open('XGBClassifier.pkl', 'rb') 
        carga_modelo = pickle.load(open('XGBC.pkl', 'rb'))
        transformer = carga_transformer
        modelo = carga_modelo
        df = transformer.transform(input_df)
        st.write(df)
        predict=modelo.predict(df)
        #print (modelo)
        result = (predict) 
     
        st.success('el resultado de su prueba es:  {}'.format(result))




if __name__=='__main__': 
    main() 