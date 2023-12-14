# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 19:39:29 2023

@author: Scofield
"""
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler as scaler

loaded_model = pickle.load(open('C:/Users/Scofield/MACHINE LEARNING PROJECTS/Classification of Dry Beans/Bean_classification_model.sav', 'rb')) # rb means read binaryloaded_model = pickle.load(open('C:/Users/Scofield/diabetes_ai_model.sav', 'rb')) # rb means read binary

# Creating afunction for prediction

def dry_beans_category_prediction(input_data):
    # changing the input data into numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    # Reshape the array as we are predicting for one instance
    reshaped_input_data = input_data_as_numpy_array.reshape(1, -1)

    standardized_data = scaler.transform(reshaped_input_data)

    prediction = loaded_model.predict(standardized_data)

    print(prediction)

    if prediction == prediction:
        return 'This bean is' + prediction

        
def main():
    # Give the web app a title
    st.title('Dry Beans Classification Web App')
    # Getting the input data from the user
    Area = st.text_input('Bean Area')
    Perimeter = st.text_input('Bean Perimeter')
    MajorAxisLength = st.text_input('Bean MajorAxisLength')
    MinorAxisLength = st.text_input('Bean MinorAxisLength')
    ConvexArea = st.text_input('Bean ConvexArea')
    EquivDiameter = st.text_input('Bean EquivDiameter')
    roundness= st.text_input('Bean Roundness')
    ShapeFactor1 = st.text_input('Bean ShapeFactor1')
    
         
   # Code for prediction
    bean_category_test= ''
   
   # creating a prediction button
   
    if st.button('Bean Test Result'):
       bean_category_test = dry_beans_category_prediction([Area, Perimeter, MajorAxisLength, MinorAxisLength, ConvexArea, EquivDiameter, roundness, ShapeFactor1 ])

    st.success(bean_category_test)
   
   
   
   
   
if __name__=='__main__':
    main()
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   