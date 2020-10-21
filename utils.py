import pandas as pd
import numpy as np
import json
import streamlit as st

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


def find_category(x):
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




