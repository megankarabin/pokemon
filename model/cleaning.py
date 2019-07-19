import numpy as np
import pandas as pd
from sklearn_pandas import DataFrameMapper, CategoricalImputer
from sklearn.preprocessing import LabelBinarizer
from sklearn.impute import SimpleImputer

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


# df


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

zf
