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

data = pd.read_excel('C:\\Users\\natal\\Desktop\\Studia\\Praca inżynierska\\dane_inzynierka.xlsx',sheet_name='Arkusz1',header=0)
#data = pd.read_excel('dane_inzynierka.xlsx',sheet_name='Arkusz1',header=0)
data = data.drop(['rok wybudowania','wiek mieszkania','cena za m2'],axis=1)
print(data.head())
#X = [['powierzchnia','rynek','piętro','lokalizacja','wiek mieszkania','liczba pokoi']]
#y = [['cena']]


#data = pd.get_dummies(data)#kodowanie zero-jedynkowecoder
enc = LabelEncoder()
data['rynek'] = enc.fit_transform(data['rynek'])
data['lokalizacja'] = enc.fit_transform(data['lokalizacja'])
data['przedział wieku mieszkania'] = enc.fit_transform(data['przedział wieku mieszkania'])

X = data[['powierzchnia','rynek','piętro','lokalizacja','przedział wieku mieszkania','liczba pokoi']]
y = data[['cena']]
print(data.head())
#regresja liniowa
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)
reg = LinearRegression()
reg.fit(X_train,y_train)
predictionsreg = reg.predict(X_test)
r2_reg = r2_score(y_test,predictionsreg)
absolute_reg = mean_absolute_error(y_test,predictionsreg,multioutput='raw_values')
print(r2_reg)
print(absolute_reg)

#drzewa
tree = DecisionTreeRegressor()
#cross_val_score(tree,X,y,cv=30)
tree.fit(X_train, y_train)
predictions_tree = tree.predict(X_test)
r2_tree = r2_score(y_test,predictions_tree)
absolute_tree = mean_absolute_error(y_test,predictions_tree,multioutput='raw_values')
print(r2_tree)
print(absolute_tree)


#wielomian
poly = Pipeline([('poly', PolynomialFeatures(degree=2)),('linear', LinearRegression(fit_intercept=False))])
poly.fit(X_train, y_train)
predictions_poly = poly.predict(X_test)
score_poly = r2_score(y_test,predictions_poly)
absolute_poly = mean_absolute_error(y_test,predictions_poly,multioutput='raw_values')
print(score_poly)
print(absolute_poly)

#las losowy
forest = RandomForestRegressor(random_state=42)
forest.fit(X_train,y_train)
predictions_forest = forest.predict(X_test)
score_forest = r2_score(y_test,predictions_forest)
absolute_forest = mean_absolute_error(y_test,predictions_forest,multioutput='raw_values')
print(score_forest)
print(absolute_forest)

#tabela wyników
scores = [['Regresja liniowa',r2_reg,absolute_reg],['Regresja wielomianowa',score_poly,absolute_poly],['Drzewa decyzyjne',r2_tree,absolute_tree],
          ['Las losowy',score_forest,absolute_forest]]
df_scores = pd.DataFrame(scores)
df_scores.columns = 'Algorytm','Współczynnik determinacji','Średni błąd bezwzględny'
print(df_scores)

