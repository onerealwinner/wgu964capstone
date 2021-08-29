import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
tfidf = TfidfVectorizer(stop_words='english')

#dfRecipes = pd.read_csv('data/raw_recipes.csv')
dfRecipes = pd.read_csv('data/reducedRecipes.csv')
dfRecipes.rename(columns={'id':'recipe_id'}, inplace=True)
recipeCount = dfRecipes.shape[0]

#clean some more data specifcally for webpage
dfRecipes['ingredients'] = dfRecipes['ingredients'].str.replace("\'", "").str.strip('[').str.strip(']')
dfRecipes['tags'] = dfRecipes['tags'].str.replace("\'", "").str.strip('[').str.strip(']')
dfRecipes['keywords'] = dfRecipes['ingredients'] + ', ' + dfRecipes['tags']
dfRecipes['n_tags'] = dfRecipes['tags'].str.split(',').str.len()

#get some counts before filtering the data
countBreakfastType = len(dfRecipes.loc[dfRecipes['IsBreakfast'] == 1])
countLunchType = len(dfRecipes.loc[dfRecipes['IsLunch'] == 1])
countDinnerType = len(dfRecipes.loc[dfRecipes['IsMainDish'] == 1])
countSidedishType = len(dfRecipes.loc[dfRecipes['IsSideDish'] == 1])
countDessertType = len(dfRecipes.loc[dfRecipes['IsDessert'] == 1])

with st.container():
    st.sidebar.subheader('Recipes Filters')

    #mealTime = st.sidebar.radio(
    mealTime = st.sidebar.selectbox(
        "Which meal are you looking to plan for?",
        ('Dinner', 'Lunch', 'Breakfast', 'Side dish', 'Dessert'))

    mealType = st.sidebar.selectbox(
        "Are you looking for something healthy, meat, vegetarian?",
        ('Any', 'Healthy', 'Meat', 'Vegetarian'))

    st.sidebar.write("Exclude any of these allergiens?")
    excludeEggs = st.sidebar.checkbox('Exclude Eggs Or Dairy')
    excludeNuts = st.sidebar.checkbox('Exclude Nuts')
    excludeShellfish = st.sidebar.checkbox('Exclude Shellfish')

    filterType = st.sidebar.radio(
        "What filter for recommendations would you like to use?",
        ('Count Vector', 'TF-IDF'))

#Reduce by options
if(mealTime == 'Dinner'):
    dfRecipes = dfRecipes.loc[dfRecipes['IsMainDish'] == 1]

if(mealTime == 'Lunch'):
    dfRecipes = dfRecipes.loc[dfRecipes['IsLunch'] == 1]

if(mealTime == 'Breakfast'):
    dfRecipes = dfRecipes.loc[dfRecipes['IsBreakfast'] == 1]

if(mealTime == 'Side dish'):
    dfRecipes = dfRecipes.loc[dfRecipes['IsSideDish'] == 1]

if(mealTime == 'Dessert'):
    dfRecipes = dfRecipes.loc[dfRecipes['IsDessert'] == 1]

if(mealType == "Meat"):
    dfRecipes = dfRecipes.loc[dfRecipes['IsMeatDish'] == 1]

if(mealType == "Vegetarian"):
    dfRecipes = dfRecipes.loc[dfRecipes['IsVegetarian'] == 1]

if(mealType == "Healthy"):
    dfRecipes = dfRecipes.loc[dfRecipes['IsHealthy'] == 1]

if(excludeEggs):
    dfRecipes = dfRecipes.loc[dfRecipes['HasEggsOrDairy'] == 0]

if(excludeNuts):
    dfRecipes = dfRecipes.loc[dfRecipes['HasNuts'] == 0]

if(excludeShellfish):
    dfRecipes = dfRecipes.loc[dfRecipes['HasShellfish'] == 0]

#reset index
dfRecipes.reset_index(drop=True, inplace=True)

#######
recipesCount = dfRecipes.shape[0]

#show tags
ingredientsOverview = dfRecipes['ingredients'].str.strip('[').str.strip(']').str.split(', ')
ingredientsOverview = ingredientsOverview.explode().value_counts().rename_axis('ingredient').reset_index(name='counts')
ingredientsOverview["percentRecipes"] = ingredientsOverview["counts"] / recipeCount
ingredientsCount = len(ingredientsOverview)

tagsOverview = dfRecipes['tags'].str.strip('[').str.strip(']').str.split(', ')
tagsOverview = tagsOverview.explode().value_counts().rename_axis('tag').reset_index(name='counts')
tagsOverview["percentRecipes"] = tagsOverview["counts"] / recipeCount
tagsCount = len(tagsOverview)

### Really the start of the page content....
st.write("""We have """, recipesCount, "recipes")

st.write("""These use """, ingredientsCount, "ingredients and have", tagsCount, "descriptive tags")

st.subheader('pick a recipe')
recipe = st.selectbox('recipes', dfRecipes.sort_values(by='name', ascending=True)['name'])

#now show the recommendation
recipe_index = dfRecipes[dfRecipes['name'] == recipe].index.values[0]

#Recommendations by vector count
if (filterType == "Count Vector"):
    recipeName = recipe
    number_of_hits=6
    text = dfRecipes.keywords.tolist()
    vectorizer = CountVectorizer(text)
    vectors = vectorizer.fit_transform(text).toarray()

    recipe_index = dfRecipes[dfRecipes['name'] == recipeName].index.values[0]

    cosines = []
    for i in range(len(vectors)):
        vector_list = [vectors[recipe_index], vectors[i]]
        cosines.append(cosine_similarity(vector_list)[0,1])

    cosines = pd.Series(cosines)
    scoresListed = pd.DataFrame(cosines)
    
    a1 = scoresListed.stack().reset_index()
    a1.columns = ['remove', 'cosIndex','sim']
    a1 = a1.sort_values(by="sim", ascending=False)
    a1 = a1.drop(columns=['remove', 'cosIndex'])
    a1["Matching"] = round(a1["sim"]*100,2)
    a1["Matching"] = a1["Matching"].astype('str') + '%'
    a1 = a1.drop(columns=['sim'])

    #matches = dfRecipes.loc[index]
    matches = a1.merge(dfRecipes, left_index=True, right_index=True, how="inner")
    matches = matches.drop(columns=['recipe_id', 'contributor_id','submitted'
        ,'nutrition','steps','description','ingredients', 'IsMainDish', 'IsMeatDish'
        , 'IsVegetarian', 'IsDessert', 'IsHealthy','IsSideDish', 'HasEggsOrDairy', 'HasNuts', 'HasShellfish','IsLunch', 'IsBreakfast'])
    st.write('Count Vector Recommendations',matches[1:6])
    #st.write('matches len', len(matches))

#end if CountVector method

#Recommendations by tfidf
if (filterType == "TF-IDF"):
    recipe_name = recipe
    dfRecipes['keywords'] = dfRecipes['keywords'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dfRecipes['keywords'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(dfRecipes.index, index=dfRecipes['recipe_id']).drop_duplicates()

    recipe_id = dfRecipes.loc[dfRecipes['name'] == recipe_name].recipe_id
    idx = indices[recipe_id]

    sim_scores = cosine_sim[idx]

    scoresListed = pd.DataFrame(cosine_sim[idx])

    a1 = scoresListed.stack().reset_index()
    a1 = a1.sort_values(by=0,ascending=False)
    a1.columns = ['remove', 'cosIndex','sim']
    a1 = a1.drop(columns=['remove', 'cosIndex'])
    a1["Matching"] = round(a1["sim"]*100,2)
    a1["Matching"] = a1["Matching"].astype('str') + '%'
    a1 = a1.drop(columns=['sim'])
    matches = a1[1:6]

    matches = matches.merge(dfRecipes, left_index=True, right_index=True, how="inner")
    matches = matches.drop(columns=['recipe_id', 'contributor_id','submitted'
        ,'nutrition','steps','description','ingredients', 'IsMainDish', 'IsMeatDish'
        , 'IsVegetarian', 'IsDessert', 'IsHealthy','IsSideDish', 'HasEggsOrDairy', 'HasNuts', 'HasShellfish','IsLunch', 'IsBreakfast'])

    #dfRecipes = dfRecipes.merge(dfReviewsGroup, left_on='id', right_on='recipe_id', how='inner')
    st.write('TF-IDF Recommendations')
    st.dataframe(matches)

#end if TFIDF Method


#components.html("""<hr style="height:3px;margin:3px;padding:3px;border:none;color:#333;background-color:#333;" /> """)
st.write('')
st.subheader("Data overview")

sns.set(rc={'figure.figsize':(26,6)})
sns.set(rc={'xtick.labelsize':14})


st.sidebar.write("This data set contains")
st.sidebar.write('Main Dishes', countDinnerType)
st.sidebar.write('Lunches', countLunchType)
st.sidebar.write('Breakfasts', countBreakfastType)
st.sidebar.write('Side dishes', countSidedishType)
st.sidebar.write('Desserts', countDessertType)

st.subheader("Ingredients")
df2 = ingredientsOverview[:19]
fig, ax = plt.subplots()
sns.barplot(x="ingredient", y="counts", data=df2, ax=ax)
plt.xticks(rotation=45, ha='right')
st.write(fig)

st.subheader("Tags")
df1 = tagsOverview[:19]
fig, ax = plt.subplots()
sns.barplot(x="tag", y="counts", data=df1, ax=ax)
plt.xticks(rotation=45, ha='right')
st.write(fig)

st.subheader("Number of keywords per recipe")
#st.write("*Keywords help the algorithm identify matches")
numOfKeywords = []
for keyword in dfRecipes['keywords']:
    numOfWords = len(keyword.split(','))
    numOfKeywords.append(numOfWords)

fig, ax = plt.subplots()
sns.histplot(data=numOfKeywords, ax=ax, kde=True)
st.write(fig)

st.subheader("Correlations 'Time - Steps - Ingredients - Tags'")
dfCorr = dfRecipes[['minutes', 'n_steps', 'n_ingredients', 'n_tags']].corr(method="pearson")
fig, ax = plt.subplots()
sns.heatmap(data=dfCorr, ax=ax)
st.write(fig)

st.subheader("Correlations - meal types")
#st.write("we can see little correlation here, by it's very little")
dfCorr = dfRecipes[['IsMainDish','IsLunch','IsBreakfast','IsSideDish','IsDessert']].corr(method="pearson")
fig, ax = plt.subplots()
sns.heatmap(data=dfCorr, ax=ax)
st.write(fig)

st.subheader("Correlations - selected meal type to nutrional values")
dfCorr = dfRecipes[['calories','fat','sugar','sodium','protein','saturated fat','carbs']].corr(method="pearson")
fig, ax = plt.subplots()
sns.heatmap(data=dfCorr, ax=ax)
st.write(fig)


# if (filterType == "Count Vector"):
#     hmData = pd.DataFrame(cosines[:50,:50])
#     hmData = hmData
#     fig, ax = plt.subplots()
#     sns.heatmap(data=hmData, ax=ax)
#     st.write(fig)

if (filterType == "TF-IDF"):
    #HeatMap cosine_sim
    st.subheader("Correlations of recipes - preview of 50 entries")
    hmData = pd.DataFrame(cosine_sim[:50,:50])
    hmData = hmData
    fig, ax = plt.subplots()
    sns.heatmap(data=hmData, ax=ax)
    st.write(fig)


#dfRecipeStats= dfRecipes[['minutes', 'rating_average', 'rating_count', 'calories', 'n_steps', 'n_ingredients']]
#fig, ax = plt.subplots()
#sns.histplot(data=dfRecipeStats, ax=ax)
#st.write(fig)
