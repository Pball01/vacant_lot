from email.policy import default
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import time, datetime,date
import pickle
import sklearn
import geopandas as gpd

from shapely.geometry import Point, Polygon #for mapping
import seaborn as sns

#for models
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from xgboost import XGBClassifier

from joblib import dump, load
import joblib
import eli5

plt.style.use('seaborn')



###############################################################################################
#Model Part

@st.cache(suppress_st_warning=True)
def model_data(parcel_no):
    df_model1 = pd.read_csv('data_for_model.csv', index_col = 0).reset_index(drop = True)
    df_model = df_model1.copy()
    df_model = df_model.loc[df_model['parcel_number_assess'].isin(parcel_no)].reset_index(drop=True)

    df_model['zip_code_assess'] = df_model['zip_code_assess'].astype('category')
    df_model['TRACTCE10_acs'] = df_model['TRACTCE10_acs'].astype('category')
    df_model['BLKGRPCE10_acs'] = df_model['BLKGRPCE10_acs'].astype('category')
    df_model['GEOID10_acs'] = df_model['GEOID10_acs'].astype('category')
    #df['vacant_assess'] = df['vacant_assess'].astype('category')

    df_model['is_actionable_tax'] = df_model['is_actionable_tax'].astype('category')
    df_model['sequestration_enforcement_tax'] = df_model['sequestration_enforcement_tax'].astype('category')
    df_model['payment_agreement_tax'] = df_model['payment_agreement_tax'].astype('category')
    df_model['building_category_tax'] = df_model['building_category_tax'].astype('category')
    df_model['sheriff_sale_tax'] = df_model['sheriff_sale_tax'].astype('category')


    #replacing null values with 0 for all integer columns
    df_model['num_years_owed_tax'] = df_model['num_years_owed_tax'].fillna(0)
    df_model['total_due_tax'] = df_model['total_due_tax'].fillna(0)
    df_model['casenumber_diff_vio'] = df_model['casenumber_diff_vio'].fillna(0)
    df_model['violationcode_diff_vio'] = df_model['violationcode_diff_vio'].fillna(0)
    df_model['num_vacant_code_vio'] = df_model['num_vacant_code_vio'].fillna(0)

    inputs = df_model.drop(['parcel_number_assess'], axis = 1)
    return inputs

classifier = pickle.load(open("xgbclassifier.pkl", 'rb'))

dict = {'year_built_assess_diff' : 'Age of Property', 
        'market_value_2015_assess' : 'Market Value of Property in 2015', 
        'zip_code_assess_19123.0' : 'Zip Code = 19123', 
        'payment_agreement_tax_False' : 'Did not have tax payment agreement',
        'zip_code_assess_19147.0' : 'Zip Code = 19147', 
        'zip_code_assess_19132.0' : 'Zip Code = 19132', 
        'GEOID10_acs_421010013005' : 'Census GEOID = 421010013005', 
        'market_value_2021_assess' : 'Market Value of Property in 2021',
        'zip_code_assess_19130.0' : 'Zip Code = 19130', 
        'zip_code_assess_19122.0' : 'Zip Code = 19122'}

def top_feature(parcel_no, classifier):
    numeric_features = model_data(parcel_no).select_dtypes(exclude = 'category').columns
    categorical_features = model_data(parcel_no).select_dtypes(include = 'category').columns

    onehot_columns = list(classifier.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(input_features= categorical_features))

    numeric_features_list = list(numeric_features)
    numeric_features_list.extend(onehot_columns)

    classifier.named_steps['classifier'].get_booster().feature_names = numeric_features_list
    imp = classifier.named_steps['classifier'].get_booster().get_score(importance_type='gain')
    imp_df = pd.DataFrame(imp.items())
    imp_df = imp_df.rename(columns ={0 : "Reasons", 1: "gain"})
    imp_df = imp_df.sort_values(ascending = False, by = "gain")
    imp_df = imp_df.replace({"Reasons": dict})
    top_list = imp_df.head(10)['Reasons'].reset_index(drop = True)
    return top_list


def prediction(parcel_no):
    # Making predictions 
    prediction = classifier.predict(model_data(parcel_no))
    if prediction == 1:
        pred = 'VACANT LOT'
    else:
        pred = 'NOT VACANT LOT'
    return pred


def main():
    st.title("Predicting Vacant Lots in Philadelphia")


    df2 = pd.read_csv('acs_city.csv', usecols =['parcel_number_assess','location_assess', 'lat_assess', 'lng_assess', 'zip_code_assess']).reset_index(drop = True)

    #selecting addess
    option = st.multiselect('What is the address of the place? Please select one address only. ', 
                        options = list(df2['location_assess'].unique()), 
                        #default = ['2550 CASTOR AVE', '2725 E BUTLER ST'])
                        default = ['2550 CASTOR AVE'])

    check =  any(item in option for item in list(df2['location_assess'].unique()))

    
    if check is True:
        st.write('We found the address {} 😊!'.format(option[0]))
    else:
        st.write("Address not found {} 😔".format(option[0]))
        exit()


    parcel_no = df2.loc[df2['location_assess'].isin(option)]['parcel_number_assess'].unique()
    #selecting parcel number
    st.write("The following parcel numbers will be considered for the model:", pd.DataFrame({
                                                                                        'Parcel Number': parcel_no,
                                                                                        'Address': option,
                                                                                        'Zip Code': df2.loc[df2['location_assess'].isin(option)]['zip_code_assess'].astype('category')}))

#plotting location
    crs = {'init': 'epsg:4326'}
    geometry = [Point(xy) for xy in zip(df2["lat_assess"], df2["lng_assess"])]
    df2 = gpd.GeoDataFrame(df2,
                            crs = crs,
                            geometry = geometry)


    st.write("The location of the address is: ")
    #plotting with zip codes
    poly_zip = gpd.read_file("data/zip_shape/Zipcodes_Poly-shp/Zipcodes_Poly.shp")# uploading dataset
    fig,ax = plt.subplots(figsize =(15,15))
    plt.title("Selected Address in Zip Code")
    poly_zip.to_crs(epsg = 4326).plot(ax = ax, color = "white", edgecolor='black')
    df2[df2['parcel_number_assess'].isin(parcel_no)].plot(ax = ax, color = "red", marker = "*", markersize=200, label = "Selected Address")
    plt.legend(prop = {'size' : 15})
    #plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(parcel_no) 
        st.success('Your address is {}'.format(result))
        reasons = top_feature(parcel_no, classifier)
        st.success("Top 10 reason impacting this result are:")
        st.write(reasons)


    
if __name__=='__main__': 
    main()

