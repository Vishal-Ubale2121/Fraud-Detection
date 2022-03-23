import joblib
from nltk.tokenize.treebank import TreebankWordDetokenizer as wd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import pycountry as pyc
import datetime as dt

model = joblib.load("CFD_ML/algorithms/model.joblib")
wi_fn = joblib.load("CFD_ML/algorithms/first_name.joblib")
wi_ln = joblib.load("CFD_ML/algorithms/last_name.joblib")
wi_coo = joblib.load("CFD_ML/algorithms/country_of_origin.joblib")
wi_cor = joblib.load("CFD_ML/algorithms/country_of_residence.joblib")


class CustomerFraudDetection:
    # Complete processing of Machine Learning model
    # Identify New Customers
    def __init__(self, input_data):
        self.input_data = input_data

    def new_customer_identification(self, input_data):
        fname_model_list = list(wi_fn.keys())
        lname_model_list = list(wi_ln.keys())
        input_data["First_Name"] = input_data["First_Name"].str.lower()
        input_data["Last_Name"] = input_data["Last_Name"].str.lower()
        input_data['Dedup'] = input_data.First_Name.isin(fname_model_list).astype(int)
        input_data['Dedup'] = input_data.Last_Name.isin(lname_model_list).astype(int)

        return input_data

        # Data Pre processing

    def preprocessing(self, input_data):
        input_data[['DD', 'MM', 'YYYY']] = input_data.DOB.str.split("-", expand=True, )
        input_data['DD'] = input_data['DD'].astype(int)
        input_data['MM'] = input_data['MM'].astype(int)
        input_data['YYYY'] = input_data['YYYY'].astype(int)
        # Date_of_joining
        input_data[['DDj', 'MMj', 'YYj']] = input_data.Date_of_joining.str.split('-', expand=True)
        input_data[['DDj', 'MMj', 'YYj']] = input_data[['DDj', 'MMj', 'YYj']].astype(int)

        # Date_of_exit
        input_data[['DDe', 'MMe', 'YYe']] = input_data.Date_of_exit.str.split('-', expand=True)
        input_data[['DDe', 'MMe', 'YYe']] = input_data[['DDe', 'MMe', 'YYe']].astype(int)

        # Categorical Values
        input_data.Deceased_Flag = input_data.Deceased_Flag.map({True: 1, False: 0})
        input_data.Gender = input_data.Gender.map({'M': 1, 'F': 0})
        input_data.Martial_Status = input_data.Martial_Status.map({'Married': 1, 'Not Married': 0})
        input_data.PEP_Flag = input_data.PEP_Flag.map({True: 1, False: 0})
        input_data.CTF_Flag = input_data.CTF_Flag.map({True: 1, False: 0})

        # COO and COR
        input_COR = list(input_data['Country_of_residence'])
        input_COO = list(input_data['Country_of_Origin'])
        countries = {}
        for country in pyc.countries:
            countries[country.name] = country.alpha_2

        COR = [countries.get(country, 'RU') for country in input_COR]
        COO = [countries.get(country, 'RU') for country in input_COO]

        input_data['COR'] = COR
        input_data['COO'] = COO

        # Now DOB column can be dropped from the dataframe
        input_data = input_data.drop(
            columns=['DOB', 'Identifier', 'Date_of_joining', 'Date_of_exit', 'Country_of_residence',
                     'Country_of_Origin', 'Product_name', 'Risk_level'])

        input_data = self.new_customer_identification(input_data)

        try:
            input_data = input_data.replace({"First_Name": wi_fn})
            input_data = input_data.replace({"Last_Name": wi_ln})
            input_data = input_data.replace({"COO": wi_coo})
            input_data = input_data.replace({"COR": wi_cor})

        except Exception:
            return {"status": "Error", "message": "Error in conversion"}

        # cols = list(input_data.columns)
        # cols = [cols[-1]] + cols[:-1]
        # input_data=input_data[cols]
        # st.write(input_data)

        return input_data

    # user_input_pr=preprocessing(user_input)
    def predict(self, input_data):
        return model.predict_proba(input_data)

    def postprocessing(self, input_data):
        if input_data[1] <= 0.4:
            label = 'False Positive'
        else:
            label = 'Suspicious'
        return {"probability": input_data[1], "label": label, "status": "OK"}

    def compute_prediction(self):
        actual_data = self.input_data
        input_data = self.preprocessing(self.input_data)
        pred_full = {}
        for i in input_data.index:
            # st.write(predict(input_data[i:i+1]))
            if input_data.at[i, 'Dedup'] == 1:
                prediction = self.predict(input_data.iloc[i:i + 1, :-1])[0]  # for the complete file
                # st.write("Prediction is", prediction)
                prediction = self.postprocessing(prediction)
                # st.write("Prediction post processing is", prediction)
                pred_full.update({i: prediction['label']})
            else:
                label = 'New Customer'
                prediction = {"probability": 2, "label": label, "status": "OK"}
                pred_full.update({i: prediction['label']})

        df_pred = pd.DataFrame(list(pred_full.items()), columns=['id', 'label'])

        df_pred.set_index('id', inplace=True)
        df_pred.reset_index(inplace=True)

        input_data['label'] = df_pred['label']
        input_data['First_Name'] = actual_data['First_Name']
        input_data['Last_Name'] = actual_data['Last_Name']
        input_data['COR'] = actual_data['Country_of_residence']
        input_data.rename(columns={'COR': 'Country_of_residence'}, inplace=True)
        input_data[['Product_name', 'Risk_level']] = actual_data[['Product_name', 'Risk_level']]
        input_data['DateTime'] = str(dt.datetime.today())
        input_data.drop(
            columns=['DDj', 'DDe', 'MMj', 'MMe', 'YYj', 'YYe', 'Dedup', 'Customer_Type', 'Deceased_Flag', 'Gender',
                     'Martial_Status', 'PEP_Flag', 'CTF_Flag'
                , 'COO'], inplace=True)
        return input_data

    # Lookup Single User Data

    # Data Pre processing
    def lookup_preprocessing(self, input_data):
        # JSON to pandas DataFrame
        input_data = pd.DataFrame(input_data, index=[0])
        input_data["First_Name"] = input_data["First_Name"].str.lower()
        input_data["Last_Name"] = input_data["Last_Name"].str.lower()
        input_data = input_data.replace({"First_Name": wi_fn})
        input_data = input_data.replace({"Last_Name": wi_ln})

        # DOB to be split into DD MM and YYYY for ML algo
        input_data[['DD', 'MM', 'YYYY']] = input_data.DOB.str.split("-", expand=True, )
        # Now DOB column can be dropped from the dataframe
        input_data = input_data.drop(columns='DOB')
        input_data['DD'] = input_data['DD'].astype(int)
        input_data['MM'] = input_data['MM'].astype(int)
        input_data['YYYY'] = input_data['YYYY'].astype(int)

        return input_data

    # user_input_pr=preprocessing(user_input)
    def lookup_predict(self, input_data):
        return model.predict_proba(input_data)

    def lookup_postprocessing(self, input_data):
        if input_data[1] == 1:
            label = 'Suspicious'
        else:
            label = 'Not Suspicious'
        return {"probability": input_data[1], "label": label, "status": "OK"}

    def lookup_compute_prediction(self):
        try:
            input_data = self.lookup_preprocessing(self.input_data)
            # st.write(input_data)
            prediction = self.lookup_predict(input_data)[0]  # only one sample
            prediction = self.lookup_postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction