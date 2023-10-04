import os
import sys
import pandas as pd
import numpy as np
import streamlit as st 
import pickle
from src.pipline.predict_pipline import PredictPipline


teams = [
"Afghanistan",
"Australia",
"Bangladesh",
"England",
"India",
"Ireland",
"Namibia",
"Netherlands",
"New Zealand",
"Oman",
"Pakistan",
"Papua New Guinea",
"Scotland",
"South Africa",
"Sri Lanka",
"United Arab Emirates",
"West Indies",
"Zimbabwe",
]

st.title("T20 1st Inning Score Predictor")

col1,col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))

with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(teams))


col3,col4,col5 = st.columns(3)

with col3:
    current_score = st.number_input("Current Score")

with col4:
    overs = st.number_input("Overs Done(Works for over>5)")

with col5:
    wickets = st.number_input("Wickets Out")

last_five_overs = st.number_input("Runs Scored In Last 5 Overs")

if st.button("Predict Score"):
    balls_left = 120 - (overs*6)
    wickets_left = 10 - wickets
    current_run_rate = current_score/overs


    input_data = pd.DataFrame(
    {"batting_team":[batting_team],"bowling_team":[bowling_team],"current_score":[current_score],"balls_left":[balls_left],"wickets_left":[wickets],"current_run_rate":[current_run_rate],"last_five_overs":[last_five_overs]})

    predict_pipline = PredictPipline()

    result = predict_pipline.prediction(input_data)

    st.header("Predicted Score - " + str(int(result[0])))