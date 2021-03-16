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
features = df.drop(columns=[25]).to_numpy()
result = df[25].to_numpy()


#pca = PCA(n_components=features_reduction, svd_solver='full')
#dim_train = pca.fit_transform(features)
regr = RandomForestRegressor()
regr.fit(features, result)



#TEST

df = pd.read_csv('warmupv4publictest.csv', index_col ='id')
df['agents'] = df['agents'].str.replace(r'\D', '')
df['agents'] = df['agents'].astype('float64')
df['altitute'] = df['altitute'].replace('average', 0.0)
df['altitute'] = df['altitute'].replace('low', -1.0)
df['altitute'] = df['altitute'].replace('high', 1.0)
imputer = KNNImputer(n_neighbors=knn_fill)
features = imputer.fit_transform(df.to_numpy())
predict = regr.predict(features)
df['sd_trans'] = predict

df['sd_trans'].to_csv('submission1.csv')