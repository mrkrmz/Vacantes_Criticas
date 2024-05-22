from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class NewDataCleaner():
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        
        # Eliminar la columna "vacant_country_name"
        X.drop("vacant_country_name", axis=1, inplace=True)
        

        # Convertir columnas categóricas a variables dummy
        X = pd.get_dummies(X, columns=["vacant_experience_and_positions", "vacant_education_level_name", "time_of_day"], drop_first=False)
        
        # Filtrar valores de salario
        return X


    def transform_y(self, y):
        return y[self.X.index]
