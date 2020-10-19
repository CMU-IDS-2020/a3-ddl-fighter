import streamlit as st
import pandas as pd
# TODO: add cov19 data package
import altair as alt

st.title("Let's analyze some yelp business data.😊")

@st.cache  # add caching so we load the data only once
def load_data_yelp():
    # Load the penguin data from https://github.com/allisonhorst/palmerpenguins.
    yelp_business_url = "https://uc38416f324414df5d8434b9cd78.dl.dropboxusercontent.com/cd/0/inline/BBiMKuC9CB9R1ZK9Y52GMkZCCcHpUNU7DJl6RhT5E8djQ_leckgrbxde6j-tlpJwqMDE0SlvohofscDF8HwXNiMoqyHdWY_ox_Pq-krmaEI4zr_RV2IQoe3-ay4_7q_g7sA/file#"
    #TODO yelp_cov19_url
    return pd.read_json(yelp_business_url, lines=True)

# load data cov19 just use your function
def load_data_cov19():
    pass

df_yelp = load_data_yelp()
df_cov19_city_level, df_cov19_state_level = load_data_cov19()

st.write("We jointly use the data from two datasets, first let us look at their dataframe seperately.")

#dataset1 dataset1pp dataset2 dataset2pp dataset1-dataset2 merge
st.write("Let us first look at the yelp business dataset.") #TODO: add some analysis
#business change
#current business situation
st.write(df_yelp.head())

#TODO: data preprocessing


st.write("Then let us look at the cov19 dataset.") #TODO: add some analysis not important
st.write(df_yelp.head())

# we know what happen



# city
# geometric not interactive



# TODO: state/city change

st.write("Hmm 🤔, is there some correlation between body mass and flipper length? Let's make a scatterplot with [Altair](https://altair-viz.github.io/) to find.")

'''
chart = alt.Chart(df).mark_point().encode(
    x=alt.X("body_mass_g", scale=alt.Scale(zero=False)),
    y=alt.Y("flipper_length_mm", scale=alt.Scale(zero=False)),
    color=alt.Y("species")
).properties(
    width=600, height=400
).interactive()


st.write(chart)
'''
