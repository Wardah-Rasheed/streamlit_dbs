import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

siteHeader = st.container()
dataExploration = st.container()
newFeatures = st.container()
modelTraining = st.container()

st.markdown(
    """
    <style>
    .main {
    background-color: #fdc3c4;
    }
    </style>
    """,
    unsafe_allow_html=True
  )



@st.cache
def get_data(filename):
    covid=pd.read_csv("WHO-COVID-19-global-table-data.csv")
    return covid

######################
# Page Title
######################


with siteHeader:
    st.title('**COVID-19: Cases and Deaths count**')
    st.text('According to WHO, as of 4 April 2022, there have been 494,587,638 confirmed cases*')


with dataExploration:
    st.header("Dataset:COVID-19's latest reported counts of cases and deaths")
    covid=get_data("WHO-COVID-19-global-table-data.csv")
    st.write(covid.head())
    
    

    st.subheader('Cases - cumulative total per 100000 population on the COVID-19 dataset')
    cum_to=pd.DataFrame(covid['Cases - cumulative total per 100000 population'].value_counts()).head(50)
    st.bar_chart(cum_to)
    st.line_chart(covid["Cases - cumulative total per 100000 population"].value_counts())

with modelTraining:
    st.header("Model Training")
    st.text("let's get started")
    # making columns
    input,display=st.columns(2)

    # pehley column main apke selection points hain
    max_depth= input.slider("What should be the max_depth of the model?", min_value=10, max_value=100, value=20, step=5)

# n_estimators:
n_estimators=input.selectbox("How many trees are there?", options=[50,100,200,300,"NO LIMIT"])

# adding list of features
input.write(covid.columns)

# Input features from users
input_features=input.text_input("Which feature you want to use ?")


# machine learning model
model=RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

#If wali baat hojaye zara
if n_estimators == "NO LIMIT" :
    model=RandomForestRegressor(max_depth=max_depth)
else:
    model=RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

# Define X and Y
X= covid[[input_features]]
y=covid[["Cases - cumulative total per 100000 population"]]

# Fit our model
model.fit(X,y)
pred=model.predict(y)

# Display metrices
display.subheader("Mean absolute error of the model is:")
display.write(mean_absolute_error(y, pred))
display.subheader("Mean squared error of the model is:")
display.write(mean_squared_error(y, pred))
display.subheader("R squared score of the model is:")
display.write(r2_score (y, pred))

 