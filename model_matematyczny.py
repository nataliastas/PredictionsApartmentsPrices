import pandas as pd
import statsmodels.api as sm
import numpy as np

data = pd.read_excel('C:\\Users\\natal\\Desktop\\Studia\\Praca inżynierska\\dane_inzynierka.xlsx',sheet_name='Rynek wtórny',header=0)
X = data[['powierzchnia','piętro','wiek mieszkania','liczba pokoi']]
y = data[['cena']]
X1 = data[['powierzchnia','wiek mieszkania']]
model = sm.OLS(y,X).fit()
model2 = sm.OLS(y,X1).fit()


data2 = pd.read_excel('C:\\Users\\natal\\Desktop\\Studia\\Praca inżynierska\\dane_inzynierka.xlsx',sheet_name='Rynek pierwotny',header=0)
X2 = data2[['powierzchnia','piętro','wiek mieszkania','liczba pokoi']]
y2 = data2[['cena']]
model3 = sm.OLS(y2,X2).fit()
X3 = data2[['powierzchnia','piętro','liczba pokoi']]
model4 = sm.OLS(y2,X3).fit()
print(model4.summary())