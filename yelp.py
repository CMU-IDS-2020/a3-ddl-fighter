import streamlit as st
import numpy as np
from datetime import date
import pandas as pd
import covidcast
import altair as alt
from wordcloud import WordCloud
import covidcast
import geopandas as gpd
import us
from geopy.geocoders import Nominatim
import addfips
from vega_datasets import data
import time
from utils import cate_list_multi, cate_list, find_category, get_highlight_info, cate_func

@st.cache  # load data from url, when submitting
def load_data_from_link():
    # load yelp business characteristic information
    yelp_business_url = "https://www.dropbox.com/s/efeev15u89g3g5p/yelp_academic_dataset_business.json?raw=1"
    yelp_business_df = pd.read_json(yelp_business_url, lines=True)
    # load yelp business covid 19 data

    yelp_covid_url = "https://www.dropbox.com/s/y6ac1lpu7ayezlj/yelp_academic_dataset_covid_features.json?raw=1"
    yelp_covid_df = pd.read_json(yelp_covid_url, lines=True)

    yelp_business_df['categories'] = yelp_business_df['categories'].fillna('')
    yelp_covid_df = yelp_covid_df.drop_duplicates(subset=['business_id'], keep='first')
    yelp_covid_df.index = range(yelp_covid_df.shape[0])

    yelp_join = pd.merge(yelp_business_df, yelp_covid_df) 

    return yelp_business_df, yelp_covid_df, yelp_join

@st.cache # load data from json, when doing local test
def load_data_from_local():
    yelp_business_df = pd.read_json("dataset/yelp_academic_dataset_business.json", lines=True)
    yelp_covid_df = pd.read_json("dataset/yelp_academic_dataset_covid_features.json", lines=True)
    
    yelp_business_df['categories'] = yelp_business_df['categories'].fillna('')
    yelp_covid_df = yelp_covid_df.drop_duplicates(subset=['business_id'], keep='first')
    yelp_covid_df.index = range(yelp_covid_df.shape[0])

    yelp_join = pd.merge(yelp_business_df, yelp_covid_df) 

    return yelp_business_df, yelp_covid_df, yelp_join

@st.cache 
#merge cov19 yelp dataset and big business dataset
def get_join_dataset(dataset1, dataset2, col):
    return pd.merge(dataset1, dataset2, on=col)

@st.cache
# get the state name to id dic
def get_state_ref():
    state_ref = pd.read_json(data.income.url)[['name', 'id']].groupby(['name']).mean()  
    # city_ref = pd.read_json("dataset/yelp_academic_dataset_covid_features.json", lines=True)
    return state_ref


@st.cache(allow_output_mutation=True)
# load data cov19 just use your function
def load_data_cov19(geo_type = 'state', start_day = date(2020, 6, 10), end_day = date(2020, 6, 10)):
    # geo_type = 'state' or 'county'
    all_measures = ['confirmed_cumulative_num', 'confirmed_cumulative_prop', 'confirmed_incidence_num', 'confirmed_incidence_prop', 'deaths_cumulative_num', 'deaths_cumulative_prop',
                'deaths_incidence_num', 'confirmed_7dav_cumulative_num', 'confirmed_7dav_cumulative_prop', 'confirmed_7dav_incidence_num', 'confirmed_7dav_incidence_prop',
                'deaths_7dav_cumulative_num', 'deaths_7dav_cumulative_prop', 'deaths_7dav_incidence_num']
    
    data_source = 'indicator-combination'
    covid_data = {}
    for measure in all_measures:
        measure_data = covidcast.signal(data_source, measure, start_day, end_day, geo_type)
        covid_data[measure] = measure_data
    
    return covid_data

@st.cache
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

@st.cache
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
                target_df[target_feature + ' type'] = [False, False, True, True]
                target_df[target_feature] = group_part
            else:
                group_part = yelp_covid_bool_df.groupby(by=[target_feature, other_feature]).agg({'business_id':'count',target_feature:'min',other_feature:'min'})
                target_df[other_feature + ' type'] = list(group_part[other_feature])
                target_df[other_feature] = list(group_part['business_id'])
        group_dict[target_feature] = target_df

    return group_dict

@st.cache
def get_category(yelp_business_df):
    business_category_info = pd.DataFrame()
    for cate in cate_list_multi:
        business_category_info[cate] = yelp_business_df['categories'].apply(cate_func[cate])
    
    return business_category_info

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
                alt.X(feature + ' type:N', title=y_title_type),
                alt.Y(other_feature + ':Q', title=x_title_type),
                alt.Color(other_feature + ' type:N'),
                alt.Tooltip([other_feature + ':Q', other_feature + ' type:N'])
            ).properties(
                width=150,
                height=150
            )
            row |= new_col
        chart &= row
    st.altair_chart(chart)
    #st.write(chart)

def show_covid_feature_multi_relationship(total_feature, yelp_covid_bool_df):
    st.write("You may also want to see how multi features affect certain feature.")
    affected = st.selectbox('Select one affected feature.', total_feature, 0)
    affecting = {}
    for other_feature in total_feature:
        if other_feature == affected:
            continue
        affecting[other_feature] = st.sidebar.selectbox(other_feature, ["Don't care", 'True', 'False'])

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

def close_for_how_long(yelp_join):
    st.markdown("From now on, we retrive the original information in **Temporary Closed Until**, **Covid Banner**, and **highlights**.")
    
    st.write("First, let's see when do these temporarly closed businesses plan to reopen in June 10. There are {} businesses uploading closure notification.".format(sum(yelp_covid_df['Temporary Closed Until'] != 'FALSE')))
    
    st.write("You may select certain category you are interested in from the bottom box, and select certain time sub-interval to take a closer look, which could be done through bruching on the upper figurg.")

    close_time = yelp_join[yelp_join['Temporary Closed Until'] != 'FALSE']['Temporary Closed Until']
    close_time = list(close_time)
    close_time = [ele[:-5] for ele in close_time]

    category = yelp_join[yelp_join['Temporary Closed Until'] != 'FALSE']['categories'].fillna('').apply(find_category)
    category = list(category)

    df = pd.DataFrame()
    df['Close Until'] = close_time
    df['Category'] = category

    brush = alt.selection_interval()
    input_dropdown = alt.binding_select(options = cate_list, name="Category of ")
    picked = alt.selection_single(encodings=["color"], bind=input_dropdown)

    base = alt.Chart(df[df['Close Until'] < '2021-01-01']).mark_area().encode(
        alt.X("Close Until:T"),
        alt.Y("count()")
    ).properties(height=50, width=500).add_selection(brush)


    chart = base & alt.Chart(df[df['Close Until'] < '2021-01-01T00:00:00']).mark_bar(size=20).encode(
        alt.X("Close Until:T",
            scale=alt.Scale(domain=brush)),
        alt.Y("count()", title='Business number'),
        alt.Tooltip(["Close Until:T", "Category:N", "count()"]),
        color = alt.condition(picked, "Category:N", alt.value("lightgray")),
    ).add_selection(picked).properties(height=300, width=500)

    st.write(chart)

def what_covid_banner_say(yelp_covid_df, business_category_info):
    feature = st.selectbox("Choose a category you are interested in: ", ['whole'] + cate_list_multi)
    
    if feature == 'whole': 
        total_num = sum(yelp_covid_df['Covid Banner'] != 'FALSE')
        banner = list(yelp_covid_df['Covid Banner'].unique())
    else:
        total_num = sum(yelp_covid_df['Covid Banner'][business_category_info[feature]] != 'FALSE')
        banner = list(yelp_covid_df['Covid Banner'][business_category_info[feature]].unique())

    cate_str = "of category {}".format(feature)
    if feature == 'whole':
        cate_str = "in all"

    st.write("Next, let's turn to Covid Banner. There are {} businesses {} uploading their Covid Banner.".format(total_num, cate_str))
    st.write("You may want see what words frequently appear in that banner, in whole, or in certain category.")
    w = WordCloud(background_color='white', width=600, height=300)
    banner_str = ' '.join(banner)
    w.generate(banner_str)
    c = w.to_image()
    st.image(c)

def what_are_highlights(business_highlight_info_short):
    st.write("Now let's take a look at hightlights, which are put in the business information page. There are 7045 businesses in all updating their highlights in response to Covid 19. We can see what highlights are about.")
    st.write("Below, we extract 11 covid 19 related highlights, and show their numbers decomposing in different categories. You may select one highlight and see in more detailed about its decomposition in the second figure. You may also select one particular category and compare highlight types of business belong to that category.")
    selected = alt.selection_multi(encodings=['y'], resolve='intersect')
    selected_cate = alt.selection_multi(encodings=['y'])
    chart = alt.Chart(business_highlight_info_short).mark_bar().encode(
        alt.X('sum(count)'),
        alt.Y('highlight:N', sort='-x'),
        alt.Tooltip(['highlight', 'sum(count)']),
        color = alt.condition(selected_cate, 'categories', alt.value("lightgray")),
    ).add_selection(selected)& alt.Chart(business_highlight_info_short).mark_bar().encode(
        alt.Y('categories:N'),
        alt.X('sum(count)'),
    ).add_selection(selected_cate).transform_filter(selected)

    st.write(chart)


def show_business_in_category(yelp_covid_bool_df, business_category_info):
    
    cat_info_cnt = pd.DataFrame()
    feature = st.selectbox("Select a Covid feature you are interested in:  ", yelp_covid_bool_df.columns[1:])

    for cate in cate_list_multi:
        cat_info_cnt[cate] = yelp_covid_bool_df[business_category_info[cate]].groupby(feature).agg({'business_id':'count'})['business_id']
    
    cat_info_cnt = cat_info_cnt.transpose()
    st.bar_chart(pd.DataFrame(
        np.array(cat_info_cnt),
        columns=['False', 'True'],
        index=cate_list_multi
    ))


yelp_business_df, yelp_covid_df, yelp_join = load_data_from_local() #load_data_from_url()

st.title("How does business behave during COVID19")

st.write("We jointly use the data from two datasets, first let us look at their dataframe seperately.")

# Part 1: Overview on yelp covid features
st.markdown("## 1. Business state overview")
st.write("Let's first look at the raw data provided by Yelp for business state under COVID19.") 

st.dataframe(yelp_covid_df.head())

# TODO: some descriptions of features
st.markdown("Bool type: **delivery or takeout**, **Grubhub enabled**, **Call To Action**, **Request a Quote Enabled**")
st.markdown("Json type: **highlights**")
st.markdown("Str type: **Covid Banner**, **Virtual Services Offered**")
st.markdown("Datetime type: **Temporary Closed Until**")

# 1.1 show the pairwise relationship between different features
st.write("These features characterize business state under COVID19. We start by looking at the inner relationship between these features.")

st.markdown("### 1.1 Covid feature correlation")
st.write("We first change non-bool type features into bool types to see whether there is some inner correlation among these features.")

yelp_covid_bool_df = get_bool_df(yelp_covid_df)
show_df = st.checkbox("Show new data")
if show_df:
    st.dataframe(yelp_covid_bool_df.head())
# TODO: maybe we can take some not FALSE value in displaying the dataset

total_feature = yelp_covid_bool_df.columns[1:]
group_dict = get_bool_df_summary(yelp_covid_bool_df)

# 1.1.1 Pairwise relationship
st.write("Now let's explore whether these features are related. You may select several interested features below and press GO to run.")
sub_feature_list = st.multiselect(
    'Show your interested subsets',
    total_feature
)
confirm_button = st.checkbox('GO!')
if confirm_button:
    show_covid_feature_relationship(group_dict, sub_feature_list)

# 1.1.2 show how one is affected by multiple
show_covid_feature_multi_relationship(total_feature, yelp_covid_bool_df)
   
#TODO: data preprocessing


st.write("Then let us look at the cov19 dataset.") 
#TODO: add some analysis not important


st.write("Let us first explore how geometric factor will influence the business situation of the yelp restaurants") 
# city
# geometric interactive, state/city
total_targets = yelp_covid_bool_df.columns[1:]
st.write("You may also want to see how is the situation for different country across the country.")
target = st.selectbox('Select one target you want to explore more.', total_targets, 0)
join_df = get_join_dataset(yelp_business_df[['business_id', 'state', 'city']], yelp_covid_bool_df, 'business_id') #only join with geometric dataset
states = alt.topo_feature(data.us_10m.url, 'states')

state_dic = join_df[['state', target]].groupby(['state']).mean()
state_ref = get_state_ref()
state_dic['id'] = -1
for row in state_dic.iterrows():
    if us.states.lookup(row[0]):
        state_name = us.states.lookup(row[0]).name
        state_dic.at[row[0], 'id'] = state_ref.loc[state_name]['id']

state_dic = state_dic[state_dic['id'] != -1]

st.write("Notice that there are some data from Canada and we remove them")

chart = alt.Chart(state_dic).mark_geoshape().encode(
    shape='geo:G',
    color=target + ':Q',
    tooltip=['state:N', target + ':Q'],
).transform_lookup(
    lookup='id',
    from_=alt.LookupData(data=states, key='id'),
    as_='geo'
).properties(
    width=600,
    height=350,
).project(
    type='albersUsa'
)
st.write(chart)


st.write("Why there are differences in business in different locations? Let's explore more about this.")
st.write("Is this because of the infection rate.")
total_targets = yelp_covid_bool_df.columns[1:]
st.write("You may also want to see how is the situation for different country across the country.")
inf_rate_target = st.selectbox('Select one target you want to explore more.', total_targets, 1)

covid_dataset = load_data_cov19('county')['confirmed_cumulative_prop']
covid_dataset['geo_value'] = covid_dataset['geo_value'].apply(lambda x: int(x))

city_dic_grouped = join_df[['city', inf_rate_target]].groupby(['city'])
city_dic = city_dic_grouped.mean()
city_dic_count = city_dic_grouped.sum()
city_dic['geo_value'] = -1
dic = pd.read_pickle('https://www.dropbox.com/s/52jh2x4p5b724c3/city_dic.pkl?raw=1')
# geolocator = Nominatim(user_agent="streamlit")
# af = addfips.AddFIPS()
for row in city_dic.iterrows():
    if city_dic_count.at[row[0], inf_rate_target] < 4:
        continue
    try:
        # addr = geolocator.geocode(row[0]).address
        # county = addr.split(", ")[1]
        # state = addr.split(", ")[2]
        # if "County" in county:
            # fips = af.get_county_fips(county, state)
            # city_dic.at[row[0], 'geo_value'] = fips 
        city_dic.at[row[0], 'geo_value'] = dic.at[row[0], 'geo_value']
    except:
        pass
city_dic = city_dic[city_dic['geo_value'] != -1]


source = get_join_dataset(city_dic, covid_dataset, 'geo_value')
st.write(source)

st.write(inf_rate_target)
base = alt.Chart(source).mark_circle().encode(
    x='value:Q',
    y=inf_rate_target + ':Q',
)
st.write(base)

st.write("## 2. Category and business state")

st.markdown("Let's now explore how businesses of different categories behave.")
business_category_info = get_category(yelp_business_df)
show_business_in_category(yelp_covid_bool_df, business_category_info)

st.markdown("### 2.1 How long do they plan to close")
close_for_how_long(yelp_join)

st.markdown("### 2.2 What do Covid Banner say")
what_covid_banner_say(yelp_covid_df, business_category_info)

st.markdown("### 2.3 What are in the highlights")
business_highlight_info = get_highlight_info(yelp_join)
what_are_highlights(business_highlight_info)


# TODO: state/city change

st.write("Hmm 🤔, is there some correlation between body mass and flipper length? Let's make a scatterplot with [Altair](https://altair-viz.github.io/) to find.")
