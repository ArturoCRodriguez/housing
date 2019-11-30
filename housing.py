import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import Imputer

def load_housing_data(housing_path):
    return pd.read_csv(housing_path)

housing = load_housing_data('housing.csv')
# se divide entre 1.5 para minimizar la extensión de la feature (pasa de 0-15 a 0-10)
# posteriormente, todo lo que sea mayor que 5 se deja en 5
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5) 
# Al hacer where income < 5 NOS ESTAMOS QUEDANDO CON LO QUE SEA MENOR QUE 5. Es decir, lo que sea menor que 5 se queda como está
# el segundo argumento indica qué se hace con lo que no cumpla la primera condición. Es decir, LOS QUE NO SEAN MENORES QUE 5 (los mayores o iguales) SE PONEN A 5.0
housing["income_cat"].where(housing["income_cat"] < 5,5.0, inplace = True)

# print(housing.describe())
# print(housing.head(100))
# housing.info()
# print(housing['ocean_proximity'].value_counts())
# housing.hist(bins = 50, figsize=(20,15))
# plt.show()
# train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)

split = StratifiedShuffleSplit(n_splits=1, test_size= 0.2 , random_state= 42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
# print(housing["income_cat"].value_counts() / len(housing))
# print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))
# print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

# al iterar sobre una dupla, vamos cogiendo cada uno de sus elementos. En este caso damos 2 vueltas. En la primera set = strat_train_set y en la segunda set = strat_test_set
for set in (strat_train_set, strat_test_set):
    # print(len(set))
    set.drop(["income_cat"], axis=1, inplace = True)

housing = strat_train_set.copy()
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha = 0.1)
# housing.plot(kind="scatter", x="longitude",y="latitude", alpha=0.4, s=housing["population"]/100, label="population",c="median_house_value", cmap=plt.get_cmap("jet"),colorbar=True)
# plt.legend()
# plt.show()
# print(housing.head(100))
# corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending = False))

# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes])
# plt.show()

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
# corr_matrix = housing.corr()
# print(housing.head(100))
# print(corr_matrix["median_house_value"].sort_values(ascending = False))
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

imputer = Imputer(strategy="median") # creamos la instancia
housing_num = housing.drop("ocean_proximity", axis=1) # Imputer solo puede trabajar con datos numéricos
imputer.fit(housing_num) # "Entrenamos" el Imputer. Simplemente calcula la mediana
# print(imputer.statistics_)
# print(housing_num.median().values)
X = imputer.transform(housing_num) # transformamos el dataset con el modelo obtenido. Es decir, en los missing values metemos la mediana de la feature
housing_tr = pd.DataFrame(X, columns= housing_num.columns) # le volvemos a poner los headers a X, ya que transform devuelve sólo las filas de datos (plain Numpy array)
# En este punto, seguimos sin "ocean_proximity"
# print(housing_tr.info())

