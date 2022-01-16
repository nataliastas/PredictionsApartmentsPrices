import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_excel('C:\\Users\\natal\\Desktop\\Studia\\Praca inżynierska\\dane_inzynierka.xlsx',
                     sheet_name='Dane',header=0)
data = data.drop(['wiek mieszkania','cena za m2'],axis=1)
X = data[['powierzchnia','liczba pokoi','piętro','lokalizacja','przedział wieku mieszkania',
          'rynek']].values
y = data[['cena']].values
print(X)

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
