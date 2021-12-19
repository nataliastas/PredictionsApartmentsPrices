import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import altair as alt

st.title('Ceny mieszkań w mieście Poznań')

@st.cache
def load_data(filename):
    data = pd.read_excel(filename,sheet_name='Dane',header=0)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data
data_load_state = st.text('Ładuję dane...')
data = load_data('C:\\Users\\natal\\Desktop\\Studia\\Praca inżynierska\\dane_inzynierka.xlsx')
data_load_state.text('Dane załadowane!')

st.header('Wszystkie dane')
if st.checkbox('Pokaż wszystkie dane'):
    st.subheader('Wszystkie dane')
    st.write(data)

st.header('Filtrowanie danych')
rynek = data['rynek'].unique().tolist()
rynek_selected = st.multiselect('Rodzaj rynku', rynek, rynek)
lokalizacja = data['lokalizacja'].unique().tolist()
lokalizacja_selected = st.multiselect('Lokalizacja', lokalizacja, lokalizacja)
pokoje = data['liczba pokoi'].unique().tolist()
pokoje_selected = st.multiselect('Liczba pokoi', pokoje, pokoje)
wiek = data['przedział wieku mieszkania'].unique().tolist()
wiek_selected = st.multiselect('Przedział wieku mieszkania', wiek, wiek)
pietro = data['piętro'].unique().tolist()
pietro_selected = st.multiselect('Numer piętra', pietro, pietro)
mask_rynek = data['rynek'].isin(rynek_selected)
mask_lokalizacja = data['lokalizacja'].isin(lokalizacja_selected)
mask_pokoje = data['liczba pokoi'].isin(pokoje_selected)
mask_wiek = data['przedział wieku mieszkania'].isin(wiek_selected)
mask_pietro = data['piętro'].isin(pietro_selected)
data_filtered = data[mask_rynek & mask_lokalizacja & mask_pokoje  & mask_wiek & mask_pietro]
st.write(data_filtered)

st.header('Wizualizacja danych')
st.text('Wybierz między jaką cechą, a ceną chcesz wyświetlić wykres zależności')
option = st.selectbox('Wybierz cechę',('powierzchnia','liczba pokoi','piętro','lokalizacja','wiek mieszkania','rynek'))
X = data[['powierzchnia','liczba pokoi','piętro','lokalizacja','wiek mieszkania','rynek']].values
y = data[['cena']].values
chart = alt.Chart(data).mark_circle().encode(x=option,y='cena')
st.altair_chart(chart)


