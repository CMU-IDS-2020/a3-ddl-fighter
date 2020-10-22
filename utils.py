import pandas as pd
import numpy as np
import json
import streamlit as st

total_covid_feature = [
    'delivery or takeout', 
    'Grubhub enabled',
    'highlights', 
    'Call To Action enabled', 
    'Request a Quote Enabled', 
    'Covid Banner',
    'Temporary Closed Until', 
    'Virtual Services Offered']

# when allowed multi categories for one business
cate_list_multi = ['Restaurants',
    'Shopping',
    'Food',
    'Home Services',
    'Beauty & Spas',
    'Health & Medical']

# when requires only unique category for one business
cate_list = ['Restaurants & Food',
    'Shopping',
    'Home Services',
    'Beauty & Spas',
    'Health & Medical']

# identifiers in highlights 
covid_19_identifier = [
    'remote_services_during_covid_19',
    'online_classes_during_covid_19',
    'takeout_during_covid_19',
    'drive_thru_during_covid_19',
    'delivery_during_covid_19',
    'curbside_pickup_during_covid_19',
    'curbside_drop_off_during_covid_19',
    'mobile_services_during_covid_19',
    'virtual_estimates_during_covid_19',
    'shipping_during_covid_19',
    'gift_cards_during_covid_19']

def if_covid_highlights(x):
    if 'covid' in x:
        return True
    return False

def if_restaurant(x):
    if 'Restaurant' in x:
        return True
    else:
        return False

def if_shopping(x):
    if 'Shopping' in x:
        return True
    else:
        return False

def if_food(x):
    if 'Food' in x:
        return True
    else:
        return False

def if_home_services(x):
    if 'Home Services' in x:
        return True
    else:
        return False

def if_beauty_spa(x):
    if 'Beauty & Spas' in x:
        return True
    else:
        return False

def if_health(x):
    if 'Health & Medical' in x:
        return True
    else:
        return False

cate_func = {cate_list_multi[0]:if_restaurant, cate_list_multi[1]:if_shopping, cate_list_multi[2]:if_food, cate_list_multi[3]:if_home_services, cate_list_multi[4]:if_beauty_spa, cate_list_multi[5]:if_health}

@st.cache
def get_category(yelp_join): # when allowed multi categories
    # business_category_info: index same as yelp_covid_df

    business_category_info = pd.DataFrame()
    for cate in cate_list_multi:
        business_category_info[cate] = yelp_join['categories'].apply(cate_func[cate])
    
    return business_category_info

def find_category(x): # when only allowed single category
    if ('Restaurant' in x) or ('Food' in x):
        return cate_list[0]
    for cat in cate_list[1:]:
        if cat in x:
            return cat
    return 'Others'

find_highlights_target = ''

def find_highlights(x):
    x_json = json.loads(x)
    for ele in x_json:
        if ele['identifier'] == find_highlights_target:
            return True
    return False

@st.cache
def get_highlight_info(df_join):
    business_highlight_info = pd.DataFrame()
    categories = df_join['categories'][df_join['highlights'] != 'FALSE']
    highlights = df_join['highlights'][df_join['highlights'] != 'FALSE']

    for target in covid_19_identifier:
        global find_highlights_target
        find_highlights_target = target

        business_highlight_info[find_highlights_target] = highlights.apply(find_highlights)

    business_highlight_info['categories'] = categories.apply(find_category)

    business_highlight_info_short = pd.DataFrame()

    attr = []
    for ele in range(len(cate_list)):
        attr += covid_19_identifier
        cat = []
        for ele in cate_list:
            cat += [ele] * len(covid_19_identifier)
        val = []
        for cate in cate_list:
            val += list(
                business_highlight_info[business_highlight_info['categories'] == cate].sum()[:-1]
    )

    business_highlight_info_short['highlight'] = attr
    business_highlight_info_short['categories'] = cat
    business_highlight_info_short['count'] = val

    return business_highlight_info_short 

@st.cache
def get_bool_df(yelp_covid_df):
    yelp_covid_bool_df = pd.DataFrame()
    for feature in yelp_covid_df.columns:
        if feature == 'business_id':
            yelp_covid_bool_df[feature] = yelp_covid_df[feature]
            continue     
        if feature == 'highlights':
            fill_in = yelp_covid_df[feature].apply(if_covid_highlights)
        else:
            fill_in = np.zeros(yelp_covid_df.shape[0], dtype='bool')
            fill_in[yelp_covid_df[feature] != 'FALSE'] = True
        yelp_covid_bool_df[feature] = fill_in

    return yelp_covid_bool_df

@st.cache
def get_bool_df_summary(yelp_covid_bool_df):
    group_dict = {}
    target_df = pd.DataFrame()
    for target_feature in total_covid_feature:
        target_df = pd.DataFrame()

        for other_feature in total_covid_feature:
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



