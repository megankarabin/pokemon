import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import pickle
from catboost import CatBoostClassifier

df = pd.read_csv('data/pokemon.csv')

train_df = pd.read_csv('data/train.csv')

res = train_df.merge(df, how='inner', left_on='First_pokemon', right_on='#')

res = res.merge(df, how='inner', left_on='Second_pokemon', right_on='#')

res['win'] = np.where(res['First_pokemon']==res['Winner'], 1, 0)

y = res['win']
X = (res.drop(columns=['win','Winner']))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

mapper = DataFrameMapper([
('First_pokemon', None),
('Second_pokemon', None),
#('#_x', None),
#('Name_x', None),
('Type 1_x', LabelBinarizer()),
#('Type 2_x', None),
('HP_x', None),
('Attack_x', None),
('Defense_x', None),
('Sp. Atk_x', None),
('Sp. Def_x', None),
('Speed_x', None),
#('Generation_x', None),
('Legendary_x', LabelBinarizer()),
#('#_y', None),
#('Name_y', None),
('Type 1_y', LabelBinarizer()),
#('Type 2_y', None),
('HP_y', None),
('Attack_y', None),
('Defense_y', None),
('Sp. Atk_y', None),
('Sp. Def_y', None),
('Speed_y', None),
#('Generation_y', None),
('Legendary_y', LabelBinarizer())
],df_out=True)

Z_train = mapper.fit_transform(X_train)
Z_test = mapper.fit_transform(X_test)

cat = CatBoostClassifier()
# log.fit(Z_train,y_train)
# log.score(Z_train,y_train)
# log.score(Z_test,y_test)
pipe = make_pipeline(mapper,cat)

pipe.fit(X_train,y_train)
pipe.score(X_test,y_test)

pickle.dump(pipe, open("pipe.pkl", "wb"))
