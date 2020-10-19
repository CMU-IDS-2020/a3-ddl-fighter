import streamlit as st
import numpy as np
from datetime import date
import pandas as pd
import covidcast
import altair as alt

st.title("How does business behave during COVID19")

@st.cache  # load data from url, when submitting
def load_data_from_link():
    # load yelp business characteristic information
    yelp_business_url = "https://uc38416f324414df5d8434b9cd78.dl.dropboxusercontent.com/cd/0/inline/BBiMKuC9CB9R1ZK9Y52GMkZCCcHpUNU7DJl6RhT5E8djQ_leckgrbxde6j-tlpJwqMDE0SlvohofscDF8HwXNiMoqyHdWY_ox_Pq-krmaEI4zr_RV2IQoe3-ay4_7q_g7sA/file#"
    yelp_business_df = pd.read_json(yelp_business_url, lines=True)
    
    # load yelp business covid 19 data

    yelp_covid_url = ''
    yelp_covid_df = pd.read_json(yelp_covid_url, lines=True)

    return yelp_business_df, yelp_covid_df

@st.cache # load data from json, when doing local test
def load_data_from_local():
    yelp_business_df = pd.read_json("dataset/yelp_academic_dataset_business.json", lines=True)
    yelp_covid_df = pd.read_json("dataset/yelp_academic_dataset_covid_features.json", lines=True)
    return yelp_business_df, yelp_covid_df

# load data cov19 just use your function
def load_data_cov19(geo_type = 'state', start_day = date(2020, 6, 10), end_day = date(2020, 6, 10)):
    # geo_type = 'state' or 'county'
    all_measures = ['confirmed_cumulative_num', 'confirmed_cumulative_prop', 'confirmed_incidence_num', 'confirmed_incidence_prop', 'deaths_cumulative_num', 'deaths_cumulative_prop',
                'deaths_incidence_num', 'confirmed_7dav_cumulative_num', 'confirmed_7dav_cumulative_prop', 'confirmed_7dav_incidence_num', 'confirmed_7dav_incidence_prop',
                'deaths_7dav_cumulative_num', 'deaths_7dav_cumulative_prop', 'deaths_7dav_incidence_num']
    
    data_source = 'indicator-combination'
    covid_data = pd.DataFrame()
    for measure in all_measures:
        measure_data = covidcast.signal(data_source, measure, start_day, end_day, geo_type)['value']
        covid_data[measure] = measure_data
    
    return covid_data

def get_bool_df(yelp_covid_df):
    yelp_covid_bool_df = pd.DataFrame()
    for feature in yelp_covid_df.columns:
        if feature == 'business_id':
            yelp_covid_bool_df[feature] = yelp_covid_df[feature]
            continue
        fill_in = np.zeros(yelp_covid_df.shape[0], dtype='bool')
        fill_in[yelp_covid_df[feature] != 'FALSE'] = True
        yelp_covid_bool_df[feature] = fill_in

    return yelp_covid_bool_df

def get_bool_df_summary(yelp_covid_bool_df):
    group_dict = {}
    total_feature = yelp_covid_bool_df.columns[1:]
    target_df = pd.DataFrame()
    for target_feature in total_feature:
        target_df = pd.DataFrame()

        for other_feature in total_feature:
            if other_feature == target_feature:
                group_part = list(yelp_covid_bool_df.groupby(by=[target_feature]).agg({'business_id':'count', target_feature:'min'})['business_id'])
                group_part.insert(1,0)
                group_part.insert(3,0)
                target_df[target_feature + '_type'] = [False, False, True, True]
                target_df[target_feature] = group_part
            else:
                group_part = yelp_covid_bool_df.groupby(by=[target_feature, other_feature]).agg({'business_id':'count',target_feature:'min',other_feature:'min'})
                target_df[other_feature + '_type'] = list(group_part[other_feature])
                target_df[other_feature] = list(group_part['business_id'])
        group_dict[target_feature] = target_df

    return group_dict

def show_covid_feature_relationship(group_dict, sub_feature_list):
    st.write("You are showing the relationship between **{}**, for each graph, the whole data points are separated into two bars according to the feature along the y-axis, and each bar is separated into two colors by the feature along the x-axis".format(', '.join(sub_feature_list)))
    chart = alt.vconcat()
    for feature in sub_feature_list:
        target_df = group_dict[feature]
        row = alt.hconcat()
        y_title_type = None
        for other_feature in sub_feature_list:
            x_title_type = None    
            if other_feature == sub_feature_list[0]:
                x_title_type = feature
            if feature == sub_feature_list[-1]:
                y_title_type = "Color: " + other_feature
            new_col = alt.Chart(target_df).mark_bar().encode(
                alt.X(feature + '_type:N', title=y_title_type),
                alt.Y(other_feature + ':Q', title=x_title_type),
                alt.Color(other_feature + '_type:N')
            ).properties(
                width=100,
                height=100
            )
            row |= new_col
        chart &= row

    st.write(chart)



yelp_business_df, yelp_covid_df = load_data_from_local() #load_data_from_url()

st.write("We jointly use the data from two datasets, first let us look at their dataframe seperately.")

# Part 1: Overview on yelp covid features
st.markdown("## Part 1")
st.write("Let's first look at the raw data provided by Yelp for business state under COVID19.") 

st.dataframe(yelp_covid_df.head())

# TODO: some descriptions of features
st.markdown("Bool type: **delivery or takeout**, **Grubhub enabled**, **Call To Action**, **Request a Quote Enabled**")
st.markdown("Json type: **highlights**")
st.markdown("Str type: **Covid Banner**, **Virtual Services Offered**")
st.markdown("Datetime type: **Temporary Closed Until**")

# 1.1 show the pairwise relationship between different features
st.write("These features characterize business state under COVID19. We start by looking at the inner relationship between these features.")

st.write("We first change non-bool type features into bool types.")

yelp_covid_bool_df = get_bool_df(yelp_covid_df)
show_df = st.checkbox("Show new data")
if show_df:
    st.dataframe(yelp_covid_bool_df.head())
# TODO: maybe we can take some not FALSE value in displaying the dataset

total_feature = yelp_covid_bool_df.columns[1:]
group_dict = get_bool_df_summary(yelp_covid_bool_df)

st.write("Now let's explore whether these features are related. You may select several interested features below and press GO to run.")
sub_feature_list = st.multiselect(
    'Show your interested subsets',
    total_feature
)
confirm_button = st.checkbox('Go!')

if confirm_button:
    show_covid_feature_relationship(group_dict, sub_feature_list)


# 1.2 show how one is affected by multiple
st.write("You may also want to see how multi features affect certain feature.")
affected = st.selectbox('Select one affected feature.', total_feature, 0)
affecting = {}
for other_feature in total_feature:
    if other_feature == affected:
        continue
    affecting[other_feature] = st.sidebar.selectbox(other_feature, ['True', 'False', "Don't care"])

confirm_button_2 = st.checkbox('Gooo!')

if confirm_button_2:
    flag = np.ones(yelp_covid_bool_df.shape[0], dtype='bool')
    for ele in affecting:
        if affecting[ele] == 'True':
            flag &= yelp_covid_bool_df[ele] == True
        elif affecting[ele] == 'False':
            flag &= yelp_covid_bool_df[ele] == False
    group_out = yelp_covid_bool_df[flag].groupby(by=[affected]).agg({'business_id': 'count', affected: 'min'})
    chart = alt.Chart(group_out).mark_bar().encode(
        alt.X(affected + ':N'),
        alt.Y('business_id:Q')
    ).properties(
        width=400,
        height=400
    )
    st.write(chart)
        
    

# st.multiselect('Select several affecting features.', total_feature)














#TODO: data preprocessing


st.write("Then let us look at the cov19 dataset.") 
#TODO: add some analysis not important



# we know what happen



# city
# geometric not interactive



# TODO: state/city change

st.write("Hmm 🤔, is there some correlation between body mass and flipper length? Let's make a scatterplot with [Altair](https://altair-viz.github.io/) to find.")

