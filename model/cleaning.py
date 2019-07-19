import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/pokemon.csv')

# df = df.reindex(labels=(range(0,152,1)))
# ty2 = []
# for x in df['Type 2']:
#     if x == np.nan:
#         ty2.append(df['Type 1'][x])
#     else:
#         ty2.append(x)
#
# df['Type 2'].isnull()


mapper = DataFrameMapper([
('#', None),
#('Name', None),
('Type 1', LabelBinarizer()),
#('Type 2', CategoricalImputer(strategy='constant', fill_value='none')),
('HP',  None),
('Attack',  None),
('Defense',  None),
('Sp. Atk', None),
('Sp. Def', None),
('Speed', None),
#('Generation', None),
('Legendary', LabelBinarizer()),
], df_out=True)

zf = mapper.fit_transform(df)

train_df = pd.read_csv('data/train.csv')


#results = zf.merge(train_df, how='inner', left_on='#', right_on='First_pokemon')

res = train_df.merge(zf, how='inner', left_on='First_pokemon', right_on='#')

res = res.merge(zf, how='inner', left_on='Second_pokemon', right_on='#')

res['win'] = np.where(res['First_pokemon']==res['Winner'], 1, 0)

y = res['win']
X = (res.drop(columns=['win','Winner']))


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

log = LogisticRegression()
log.fit(X_train,y_train)
log.score(X_train,y_train)
log.score(X_test,y_test)
