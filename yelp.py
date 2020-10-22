import streamlit as st
import numpy as np
from datetime import date
import pandas as pd
import covidcast
import altair as alt
from wordcloud import WordCloud
import covidcast
import geopandas as gpd
# import us
# from geopy.geocoders import Nominatim
# import addfips
# from vega_datasets import data
# import time
from utils import total_covid_feature, cate_list_multi, cate_list, cate_func, find_category, get_category, get_highlight_info, get_bool_df, get_bool_df_summary

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

    # merge: business_id is the order in yelp_covid_df
    yelp_join = pd.merge(yelp_covid_df, yelp_business_df, on='business_id') 

    return yelp_business_df, yelp_covid_df, yelp_join

@st.cache # load data from json, when doing local test
def load_data_from_local():
    yelp_business_df = pd.read_json("dataset/yelp_academic_dataset_business.json", lines=True)
    yelp_covid_df = pd.read_json("dataset/yelp_academic_dataset_covid_features.json", lines=True)
    
    yelp_business_df['categories'] = yelp_business_df['categories'].fillna('')
    yelp_covid_df = yelp_covid_df.drop_duplicates(subset=['business_id'], keep='first')
    yelp_covid_df.index = range(yelp_covid_df.shape[0])

    # merge: business_id is the order in yelp_covid_df
    yelp_join = pd.merge(yelp_covid_df, yelp_business_df, on='business_id') 

    return yelp_business_df, yelp_covid_df, yelp_join

@st.cache 
#merge cov19 yelp dataset and big business dataset
def get_join_dataset(dataset1, dataset2, col):
    return pd.merge(dataset1, dataset2, on=col)

'''
@st.cache
# get the state name to id dic
def get_state_ref():
    state_ref = pd.read_json(data.income.url)[['name', 'id']].groupby(['name']).mean()  
    # city_ref = pd.read_json("dataset/yelp_academic_dataset_covid_features.json", lines=True)
    return state_ref
'''



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

def show_covid_feature_summary(yelp_covid_bool_df):
    out = yelp_covid_bool_df[yelp_covid_bool_df.columns[1:]].sum(axis=0)
    yelp_covid_feature_summary = pd.DataFrame()
    yelp_covid_feature_summary['covid feature'] = out.index
    yelp_covid_feature_summary['count'] = list(out)
    yelp_covid_feature_summary['ratio'] = (yelp_covid_feature_summary['count'] / yelp_covid_df.shape[0]).round(2)

    chart = alt.Chart(yelp_covid_feature_summary).mark_bar().encode(
        alt.X('count:Q'),
        alt.Y('covid feature:N', sort='-x'),
        alt.Tooltip(['covid feature', 'ratio'])
    ).properties(width=600)

    st.write(chart)

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

def show_covid_feature_multi_relationship(total_covid_feature, yelp_covid_bool_df):
    st.write("You may also want to see how multi features affect certain feature.")
    affected = st.selectbox('Select one affected feature.', total_covid_feature, 0)
    affecting = {}
    for other_feature in total_covid_feature:
        if other_feature == affected:
            continue
        affecting[other_feature] = st.sidebar.selectbox(other_feature, ["Don't care", 'True', 'False'])

    flag = np.ones(yelp_covid_bool_df.shape[0], dtype='bool')
    for ele in affecting:
        if affecting[ele] == 'True':
            flag &= yelp_covid_bool_df[ele] == True
        elif affecting[ele] == 'False':
            flag &= yelp_covid_bool_df[ele] == False
    group_out = yelp_covid_bool_df[flag].groupby(by=[affected]).agg({'business_id': 'count', affected: 'min'})
    chart = alt.Chart(group_out).mark_bar().encode(
        alt.Y(affected + ':N'),
        alt.X('business_id:Q'),
        # TODO: add tooltip
    ).properties(
        width=600,
        height=120
    )
    st.write(chart)

def close_for_how_long(yelp_join):
    st.markdown("From now on, we retrive the original information in **Temporary Closed Until**, **Covid Banner**, and **highlights**.")
    
    st.write("First, let's see when did these temporarly closed businesses plan to reopen in June 10. There are {} businesses uploading closure notification.".format(sum(yelp_covid_df['Temporary Closed Until'] != 'FALSE')))
    
    st.write("You may select certain category you are interested in from the bottom box, and brush certain time sub-interval from the upper figure.")

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
    st.write("It is interesting that the planned reopen time is quite concentrated, most during June and July, and on the start or end of a certain month.")

def what_covid_banner_say(yelp_join, business_category_info):

    st.write("Next, let's turn to Covid Banner. You may want see what words frequently appear in that banner, in whole, or in certain category.")

    feature = st.selectbox("Choose a category you are interested in: ", ['whole'] + cate_list_multi)
    
    if feature == 'whole': 
        total_num = sum(yelp_join['Covid Banner'] != 'FALSE')
        banner = list(yelp_join['Covid Banner'].unique())
    else:
        total_num = sum(yelp_join['Covid Banner'][business_category_info[feature]] != 'FALSE')
        banner = list(yelp_join['Covid Banner'][business_category_info[feature]].unique())

    cate_str = "of category {}".format(feature)
    if feature == 'whole':
        cate_str = "in all"

    st.write("There are {} businesses {} uploading their Covid Banner.".format(total_num, cate_str))
    w = WordCloud(background_color='white', width=600, height=300)
    banner_str = ' '.join(banner)
    w.generate(banner_str)
    c = w.to_image()
    st.image(c)

    st.write("In all, the most important words and **will** and **open**. Businesses are expressing they are open and they would make special reaction towards COVID-19. Also, **appointment** and **safe/safety** are frequent, showing their concern towards health gurantee. You are encouraged to explore more into different categories.")

def what_are_highlights(business_highlight_info_short):
    st.write("Then let's take a look at **hightlights**, which are highlighted short information posted on the business home page. There are 7045 businesses in all updating their highlights in response to Covid 19. We may see what highlights are about.")
    st.write("Below, we extract 11 COVID-19 related highlights, and show their counts decomposing in different categories.")
    st.write("You may select one highlight and see in more detailed about its decomposition below.")
    st.write("You may also select one particular category from the right legend by click and compare how highlight types vary in that category.")

    selected = alt.selection_multi(encodings=['y'], resolve='intersect')
    selected_cate = alt.selection_multi(fields=['categories'], bind='legend')

    chart = alt.Chart(business_highlight_info_short).mark_bar().encode(
        alt.X('sum(count)', sort='-color'),
        alt.Y('highlight:N', sort='-x'),
        alt.Tooltip(['highlight', 'sum(count)']),
        color = alt.condition(selected_cate, alt.Color(
            'categories:N',
        ), alt.value("lightgray")),
    ).add_selection(selected_cate).add_selection(selected)& alt.Chart(business_highlight_info_short).mark_bar().encode(
        alt.Y('categories:N'),
        alt.X('sum(count)'),
        alt.Color('categories')
    ).transform_filter(selected)

    st.write(chart)
    st.write("Clearly, different categories are focus on different highlights. Reataurants and food, they are showing they support dilivery or takeout, and similarly shopping businesses provide shipping service. For service business, medical and home service focus on remote service, while beauty and spas can only send gift cards during COVID-19.")

def show_business_in_category(yelp_covid_bool_df, business_category_info):
    
    cat_info_cnt = pd.DataFrame()
    feature = st.selectbox("Select a Covid feature you are interested in: ", total_covid_feature)

    for cate in cate_list_multi:
        cat_info_cnt[cate] = yelp_covid_bool_df[business_category_info[cate]].groupby(feature).agg({'business_id':'count'})['business_id']
    
    cat_info_cnt = cat_info_cnt.stack().reset_index(level=[0,1])
    cat_info_cnt.columns = [feature, 'categories', 'count']
    chart = alt.Chart(cat_info_cnt).mark_bar().encode(
        alt.X('count'),
        alt.Y('categories:N', sort='-x'),
        alt.Color(feature+':N'),
        alt.Tooltip(['count', feature]),
        order = alt.Order(
            feature,
            sort='descending'
        )
    ).properties(height=200, width=700)
    st.altair_chart(chart)

    st.write("Of course, different types of businesses react differently. For example, you can see a huge part of Reastaurants and Food support **delivery and takeout**, while **request a quote** is nearly exclusively supported by Home Services.")
    
def show_quality_summary(yelp_join):
    st.write("We first look at the distribution of **star ratings** and **review counts** in our dataset. Recall that the quality data is collected in March 2020, when the COVID was not that a serious concern, and thus could be used as a previous quality measure.")

    st.write("Brush certain intervals from both figures to explore into certain stars and/or review counts.")

    out_start = yelp_join.groupby(['stars']).agg({'business_id':'count'})
    df_star = pd.DataFrame()
    df_star['stars'] = out_start.index
    df_star['count'] = list(out_start['business_id'])

    bins = list(range(0, 150, 10))
    labels = list(range(5, 145, 10))
    out_review = pd.cut(yelp_join['review_count'], bins=bins, labels=labels)
    out_review = pd.DataFrame(out_review, dtype='object')
    out_review['stars'] = yelp_join['stars']

    out_df = out_review.groupby(['stars', 'review_count']).agg({'stars':'count'}).unstack().fillna(0).stack()    
    out_df.columns = ['count']
    out_df = out_df.reset_index(level=[0,1])

    select = alt.selection_interval(encodings=['x'], resolve='intersect')

    chart = alt.Chart(out_df).mark_bar(size=15).encode(
                alt.X(
                    alt.repeat("column"),
                    type='quantitative',
                ),
                alt.Y('sum(count)', title='count')
            ).properties(height=200, width=250)
    combine_chart = alt.layer(
                    chart.add_selection(select).encode(
                        color = alt.value('lightgray')
                    ),
                    chart.transform_filter(select)
                ).repeat(
                    column=['stars', 'review_count']
                )
    st.write(combine_chart)
    st.markdown("You may find that **stars** are relatively scattered, while for **review counts**, they are pretty concentrated, and actually, though the most popular business can have more than 10,000 reviews, the 95 percentile of review count is 145.")

def quality_vs_covid_feature(yelp_covid_bool_df, yelp_join):

    st.markdown("Now, let's explore into the relationship among **stars**, **review counts** and **COVID features**. It is likely that business states are affected by their previous quality.")

    st.markdown("Let's start by selecting one **COVID feature** to see whether a business has it or not depends on stars and review counts.")

    selected_feature = st.selectbox("Select the covid feature you want to explore: ", total_covid_feature)

    st.write("Let's see how the whether the median of stars and review counts are different between businesses having {}, and those not.".format(selected_feature))

    df = pd.DataFrame()
    df['stars'] = yelp_join['stars']
    df['review_count'] = yelp_join['review_count']
    df[selected_feature] = yelp_covid_bool_df[selected_feature]

    df = df.groupby(selected_feature).agg({'stars':'median', 'review_count':'median'})
    df[selected_feature] = ['False', 'True']
    chart = alt.Chart(df).mark_bar().encode(
        alt.Y(selected_feature),
        alt.X('stars')
    ).properties(height=50, width=250) | alt.Chart(df).mark_bar().encode(
        alt.Y(selected_feature),
        alt.X('review_count')
    ).properties(height=50, width=250)
    st.write(chart)

    st.write("We can go deeper into individuals! However, due to the huge size of original dataset, we use sampling strategy. Let's choose a **sample size**!")

    sample_size = st.slider("Select how mamy points you want to sample for each case: ", min_value=50, max_value=500, value=100)

    st.markdown("You may want filter some extremely large review counts, according to your observation in the **Quality Overview**. By default, we are showing you the whole range. ")

    filter_above = 10130
    filter_above = st.number_input("I only want to sample points from review counts under: ", min_value=100, max_value=10130, value=10130)

    chk_true = yelp_covid_bool_df[selected_feature] & (yelp_join['review_count'] < filter_above)
    chk_false = (yelp_covid_bool_df[selected_feature] == False) & (yelp_join['review_count'] < filter_above)

    true_idx = np.random.choice(yelp_covid_bool_df[chk_true].index, sample_size, replace=False)
    false_idx = np.random.choice(yelp_covid_bool_df[chk_false].index, sample_size, replace=False)

    df_sample = pd.DataFrame()
    df_sample['review_count'] = list(yelp_join['review_count'][true_idx]) + list(yelp_join['review_count'][false_idx])
    df_sample['ratings'] = list(yelp_join['stars'][true_idx]) + list(yelp_join['stars'][false_idx])
    df_sample[selected_feature] = ['True'] * sample_size + ['False'] * sample_size   

    selected = alt.selection_multi(fields=[selected_feature], on='dblclick', bind='legend')
    chart = alt.Chart(df_sample).mark_point(size=35).encode(
        alt.X('review_count'),
        alt.Y('ratings:Q', scale=alt.Scale(domain=[0.5,5.5])),
        alt.Color(selected_feature),
        opacity=alt.condition(selected, alt.value(1), alt.value(0.2))
    ).add_selection(selected).properties(width=600).interactive()

    st.write("Now you have your graph! Try to double click one point or a True/False on the right to highlight a certain group.")
    st.write(chart)

    st.write("You may see that stars do not influence much, while more popular businesses are more likely to react.")
    

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
st.write("These features characterize business state under COVID19. We first change non-bool type features into bool types. Here is the summary of whether business has above covid features.")
yelp_covid_bool_df = get_bool_df(yelp_covid_df)

show_df = st.checkbox("Show transformed data")
if show_df:
    st.write(yelp_covid_bool_df.head())
    # TODO: maybe we can take some not FALSE value in displaying the dataset

show_covid_feature_summary(yelp_covid_bool_df)

st.markdown("### 1.1 Covid feature correlation")
st.write("Now let's explore the inner correlation among these features to see how these features interact with each other.")
group_dict = get_bool_df_summary(yelp_covid_bool_df)

# 1.1.1 Pairwise relationship
st.write("You may start by selecting several interested features below and press GO to run.")
sub_feature_list = st.multiselect(
    'Show your interested subsets',
    total_covid_feature
)
confirm_button = st.checkbox('GO!')
if confirm_button:
    show_covid_feature_relationship(group_dict, sub_feature_list)

# 1.1.2 show how one is affected by multiple
show_covid_feature_multi_relationship(total_covid_feature, yelp_covid_bool_df)
   
#TODO: data preprocessing

'''
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
'''

st.write("## 3. How businesses' categories affect their reaction?")

st.markdown("Let's now explore how businesses of different categories behave. We start by looking at whether different categories react differently with the above COVID features. Recall 1 represents true and 0 represents false. ")
business_category_info = get_category(yelp_join)
show_business_in_category(yelp_covid_bool_df, business_category_info)

st.markdown("### 3.1 How long do they plan to close")
close_for_how_long(yelp_join)

st.markdown("### 3.2 What do Covid Banner say")
what_covid_banner_say(yelp_join, business_category_info)

st.markdown("### 3.3 What are in the highlights")
business_highlight_info = get_highlight_info(yelp_join)
what_are_highlights(business_highlight_info)

st.write("## 4. How businesses' quality affect their reaction?")

st.markdown("Does business quality before COVID-19 have some relationship with their state during COVID-19? We would look at their popularity, measured by review counts, and their ratings.")

st.markdown("### 4.1 Quality overview")
show_quality_summary(yelp_join)

st.markdown("### 4.2 Stars, review counts, and COVID features")
quality_vs_covid_feature(yelp_covid_bool_df, yelp_join)

