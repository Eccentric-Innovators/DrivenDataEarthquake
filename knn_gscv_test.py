#%%
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

#%%
values = pd.read_csv('train_values.csv')
labels = pd.read_csv('train_labels.csv')

print(values.shape)
print(labels.shape)

#%%
X = values.drop(columns=['building_id'])
y = labels.drop(columns=['building_id'])

print("Shape of values: ", X.shape)
print("Shape of labels: ", y.shape)

#%%
categorical_columns = ['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status']
for i in categorical_columns:
	enc = OrdinalEncoder()
	X[i] = enc.fit_transform(np.array(X[i]).reshape((260601, 1)))

print(X.shape)

#%%
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1,51)}
knn_gscv = GridSearchCV(knn, param_grid, cv=5)
knn_gscv.fit(X, y)

#%%
print("Best params: ", knn_gscv.best_params_)
print("Best score: ", knn_gscv.best_score_)