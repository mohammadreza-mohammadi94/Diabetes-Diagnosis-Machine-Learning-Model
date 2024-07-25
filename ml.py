import streamlit as st
import numpy as np
import joblib
import os


ATTRIBUTE_INFORMATION = """
    - Age:  1. 20 - 65
    - Sex: 1. Male, 2.Female
    - Polyuria 1.Yes, 2.No.
    - Polydipsia 1.Yes, 2.No.
    - Sudden Weight loss 1.Yes, 2.No.
    - Weakness 1.Yes, 2.No.
    - Polyphagia 1.Yes, 2.No.
    - Genital Thrush 1.Yes, 2.No.
    - Visual Blurring 1.Yes, 2.No.
    - Itching 1.Yes, 2.No.
    - Irritability 1.Yes, 2.No.
    - Delayed Healing 1.Yes, 2.No.
    - Partial Paresis 1.Yes, 2.No.
    - Muscle Stivness 1.Yes, 2.No.
    - Alopecia 1.Yes, 2.No.
    - Obesity 1.Yes, 2.No.
    - Class 1.Positive, 2.Negative.
"""

LABEL_MAP = {"No":0,"Yes":1}
GENDER_MAP = {"Female":0,"Male":1}
TARGET_MAP = {"Negative":0,"Positive":1}

['age', 'gender', 'polyuria', 'polydipsia', 'sudden_weight_loss',
       'weakness', 'polyphagia', 'genital_thrush', 'visual_blurring',
       'itching', 'irritability', 'delayed_healing', 'partial_paresis',
       'muscle_stiffness', 'alopecia', 'obesity', 'class']


def get_fvalue(val):
	feature_dict = {"No":0,"Yes":1}
	for key,value in feature_dict.items():
		if val == key:
			return value 

def get_value(val,my_dict):
	for key,value in my_dict.items():
		if val == key:
			return value 

@st.cache_data
def load_model(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


def run_ml():
    st.subheader("Machine Learning Model")

    # Loading Modesl to application
    lr_model = load_model("models/logistic_regression_model_diabetes_21_oct_2020.pkl")

    with st.expander("Attribute Information"):
        st.markdown(ATTRIBUTE_INFORMATION)

    # Desing a layout to manage columns
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 10, 100)
        gender = st.radio('Gender', ["Female", "Male"])
        polyuria = st.radio("Polyuria",["No","Yes"])
        polydipsia = st.radio("Polydipsia",["No","Yes"]) 
        sudden_weight_loss = st.radio("Sudden Weight Loss",["No","Yes"])
        weakness = st.radio("Weakness",["No","Yes"]) 
        polyphagia = st.radio("Polyphagia",["No","Yes"]) 
        genital_thrush = st.radio("Genital Thrush",["No","Yes"]) 

    with col2:
        visual_blurring = st.radio("Visual Blurring",["No","Yes"])
        itching = st.radio("Itching",["No","Yes"]) 
        irritability = st.radio("Irritability",["No","Yes"]) 
        delayed_healing = st.radio("Delayed Healing",["No","Yes"]) 
        partial_paresis = st.radio("Partial Paresis",["No","Yes"])
        muscle_stiffness = st.radio("Muscle Stiffness",["No","Yes"]) 
        alopecia = st.radio("Alopecia",["No","Yes"]) 
        obesity = st.radio("Obesity",["No","Yes"]) 


    # Show selected options to user
    with st.expander("You've Selected:"):
        result = {'Age':age,
        'Gender':gender,
        'Polyuria':polyuria,
        'Polydipsia':polydipsia,
        'Sudden Weight Loss':sudden_weight_loss,
        'Weakness':weakness,
        'Polyphagia':polyphagia,
        'Genital Thrush':genital_thrush,
        'Visual Blurring':visual_blurring,
        'Itching':itching,
        'Irritability':irritability,
        'Delayed Healing':delayed_healing,
        'Partial Paresis':partial_paresis,
        'Muscle Stiffness':muscle_stiffness,
        'Alopecia':alopecia,
        'Obesity':obesity}

        st.dataframe(result,
                    use_container_width=True,
                    height=600)

        encoded_res = []
        for i in result.values():
            if type(i) == int:
                encoded_res.append(i)
            elif i in ["Female","Male"]:
                res = get_value(i, GENDER_MAP)
                encoded_res.append(res)
            else:
                encoded_res.append(get_fvalue(i))

    with st.expander("Prediction Results:"):
        user_info = np.array(encoded_res).reshape(1, -1)

        prediction = lr_model.predict(user_info)
        pred_prob = lr_model.predict_proba(user_info)


        if prediction == 1:
            st.warning("Positive Risk - {}".format(prediction[0]))
            pred_probability_score = {"Negative DM":pred_prob[0][0]*100,"Positive DM":pred_prob[0][1]*100}
            st.subheader("Prediction Probability Score")
            st.json(pred_probability_score)
        else:
            st.success("Negative Risk - {}".format(prediction[0]))
            pred_probability_score = {"Negative DM":pred_prob[0][0]*100,"Positive DM":pred_prob[0][1]*100}
            st.subheader("Prediction Probability Score")
            st.json(pred_probability_score)
