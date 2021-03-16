import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt

# Parametros:
knn_fill = 6
features_reduction = 10
df = pd.read_csv('warmupv4publictrain.csv')

## Converter valores para float e preencher NANs com método dos nearest neighbors
df['agents'] = df['agents'].str.replace(r'\D', '')
df['agents'] = df['agents'].astype('float64')
df['altitute'] = df['altitute'].replace('average', 0.0)
df['altitute'] = df['altitute'].replace('low', -1.0)
df['altitute'] = df['altitute'].replace('high', 1.0)
imputer = KNNImputer(n_neighbors=knn_fill)
array = imputer.fit_transform(df.to_numpy())
df = pd.DataFrame(array)

# Split de dataset entre features e resultado de treino e de teste
train, test = train_test_split(df, test_size=0.2)
train_result = train[25].to_numpy()
test_result = test[25].to_numpy()
train_features = train.drop(columns=[25]).to_numpy()
test_features = test.drop(columns=[25]).to_numpy()


#pca = PCA(n_components=features_reduction, svd_solver='full')
#dim_train = pca.fit_transform(train_features)
dim_train = train_features
regr = RandomForestRegressor()
regr.fit(dim_train, train_result)

#dim_test = pca.transform(test_features)
dim_test = test_features
predict_test = regr.predict(dim_test)

error = (predict_test - test_result)/predict_test
e = error**2
print(e.mean())

fig = plt.figure()
plt.scatter(x=predict_test, y=test_result)
plt.show()