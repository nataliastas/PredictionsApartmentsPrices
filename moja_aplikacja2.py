import streamlit as st
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_error, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error, r2_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
st.title('Przewidywanie ceny mieszkania')

#@st.cache
def load_data(filename):
    data = pd.read_excel(filename,sheet_name='Arkusz1',header=0)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data
data_load_state = st.text('Ładuję dane...')
data = load_data('C:\\Users\\natal\\Desktop\\Studia\\Praca inżynierska\\dane_inzynierka.xlsx')
data_load_state.text('Dane załadowane!')

st.header('Przewidywanie ceny')
st.text('Wybierz model jakim chcesz przewidzieć cenę')
option1 = st.selectbox('Wybierz model',('Regresja liniowa','Regresja wielomianowa','Drzewa decyzyjne','Las losowy'))
st.text('Wybierz parametry mieszkania jakiego cenę chciałbyś poznać')
enc = LabelEncoder()
data['rynek'] = enc.fit_transform(data['rynek'])
data['lokalizacja'] = enc.fit_transform(data['lokalizacja'])
data['przedział wieku mieszkania'] = enc.fit_transform(data['przedział wieku mieszkania'])

X = data[['powierzchnia','rynek','piętro','lokalizacja','przedział wieku mieszkania','liczba pokoi']]
y = data[['cena']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)
rynek = st.selectbox('Rodzaj rynku',('pierwotny','wtórny'))
lokalizacja = st.selectbox('Lokalizacja',('Stare Miasto','Nowe Miasto','Grunwald','Wilda','Jeżyce'))
powierzchnia  = st.number_input('Powierzchnia')
pokoje = st.number_input('Liczba pokoi',step=1)
pietro = st.number_input('Piętro',step=1)
if rynek=='pierwotny':
    wiek = '<0,10>'
else:
    wiek = st.selectbox('Przedział wieku mieszkania',('<0,10>','<11,30>','<31,50>','<51,70>','<71,90>','<91,110>','<111,131>'))
if rynek=='pierwotny':
    rynek=0
    wiek=0
else:
    rynek=1
if lokalizacja=='Grunwald':
    lokalizacja=0
if lokalizacja=='Jeżyce':
    lokalizacja=1
if lokalizacja=='Nowe Miasto':
    lokalizacja=2
if lokalizacja=='Stare Miasto':
    lokalizacja=3
if lokalizacja=='Wilda':
    lokalizacja=4
if wiek=='<10,0>':
    wiek=0
if wiek=='<110,91>':
    wiek=1
if wiek=='<131,111>':
    wiek=2
if wiek=='<30,11>':
    wiek=3
if wiek=='<50,31>':
    wiek=4
if wiek=='<70,51>':
    wiek=5
if wiek=='<90,71>':
    wiek=6

if st.button('Rozpocznij przewidywanie'):
#regresja liniowa
    if option1=='Regresja liniowa':
        reg = LinearRegression()
        reg.fit(X_train,y_train)
        predictionsreg = reg.predict([[powierzchnia,rynek,pietro,lokalizacja,wiek,pokoje]])
        st.write(predictionsreg)

#drzewa
    if option1=='Drzewa decyzyjne':
        tree = DecisionTreeRegressor()
        tree.fit(X_train, y_train)
        predictions_tree = tree.predict([[powierzchnia,rynek,pietro,lokalizacja,wiek,pokoje]])
        st.write(predictions_tree)

#wielomian
    if option1=='Regresja wielomianowa':
        poly = Pipeline([('poly', PolynomialFeatures(degree=2)),('linear', LinearRegression(fit_intercept=False))])
        poly.fit(X_train, y_train)
        predictions_poly = poly.predict([[powierzchnia,rynek,pietro,lokalizacja,wiek,pokoje]])
        st.write(predictions_poly)

#las losowy
    if option1=='Las losowy':
        forest = RandomForestRegressor(random_state=42)
        forest.fit(X_train,y_train)
        predictions_forest = forest.predict([[powierzchnia,rynek,pietro,lokalizacja,wiek,pokoje]])
        st.write(predictions_forest)


