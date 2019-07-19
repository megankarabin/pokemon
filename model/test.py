import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import pickle

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pd.read_csv('data/pokemon.csv')

test = pd.read_csv('data/test.csv')

tst = test.merge(df, how='inner', left_on='First_pokemon', right_on='#')
tst = tst.merge(df, how='inner', left_on='Second_pokemon', right_on='#')

y = pipe.predict(tst)

test['win'] = y

test['Winner'] = np.where(test['win']==1,test['First_pokemon'],test['Second_pokemon'])

test_pikachu = test.drop(columns='win')

test_pikachu.to_csv(r'data/test_pikachu.csv')
