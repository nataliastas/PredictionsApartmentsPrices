import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np

data = pd.read_excel('C:\\Users\\natal\\Desktop\\Studia\\Praca inżynierska\\dane_inzynierka.xlsx',sheet_name=0,header=0)
data = data.drop(['wiek mieszkania','cena za m2'],axis=1)
X = data[['powierzchnia','liczba pokoi','piętro','lokalizacja','przedział wieku mieszkania','rynek']].values
y = data[['cena']].values
print(X)
#czerwony, zielony, niebieski, fioletowy, zolty
#conv = {"Stare miasto" : [1.0,0.0,0.0], "Nowe miasto" : [0.0,1.0,0.0], "Grunwald" : [0.0,0.0,1.0], "Jeżyce" : [1.0, 0.0, 1.0], "Wilda" : [1.0,1.0,0.0]}
#local_as_numbers = []
#for i in X[:,3]:
#    local_as_numbers.append(conv[i])
plt.scatter(data['powierzchnia'],y)
plt.xlabel("Powierzchnia mieszkania")
plt.ylabel("Cena mieszkania")
plt.figure()
plt.scatter(X[:,1],y)
plt.xlabel("Liczba pokoi")
plt.ylabel("Cena mieszkania")
plt.figure()
plt.scatter(X[:,2],y)
plt.xlabel("Numer piętra")
plt.ylabel("Cena mieszkania")
plt.figure()
plt.scatter(X[:,3],y)
plt.xlabel("Lokalizacja")
plt.ylabel("Cena mieszkania")
plt.figure()
plt.scatter(X[:,4],y)
plt.xlabel("Wiek mieszkania")
plt.ylabel("Cena mieszkania")
plt.figure()
plt.scatter(X[:,5],y)
plt.xlabel("Rodzaj rynku")
plt.ylabel("Cena mieszkania")
plt.show()
#wykres korelacji
#sns.pairplot(data)
#plt.show()
#poszukiwanie korelacji
corr_matrix=data.corr()
#corr_matrix.style.background_gradient(cmap='coolwarm').set_precision(1)
#plt.matshow(corr_matrix)
#plt.show()
ax = sns.heatmap(
    corr_matrix,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)
plt.show()