#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Dataset

# 1. Download wine+quality.zip from https://archive.ics.uci.edu/dataset/186/wine+quality
# 2. Save the zip as ```data/original/wine-quality/wine+quality.zip```
# 3. ```cd data/original/wine-quality```
# 4. ```unzip wine+quality.zip```



# Citation Request:
#   This dataset is public available for research. The details are described in [Cortez et al., 2009]. 
#   Please include this citation if you plan to use this database:

#   P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
#   Modeling wine preferences by data mining from physicochemical properties.
#   In Decision Support Systems>, Elsevier, 47(4):547-553. ISSN: 0167-9236.

#   Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
#                 [Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
#                 [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib

# 1. Title: Wine Quality 

# 2. Sources
#    Created by: Paulo Cortez (Univ. Minho), António Cerdeira, Fernando Almeida, Telmo Matos and José Reis (CVRVV) @ 2009
   
# 3. Past Usage:

#   P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
#   Modeling wine preferences by data mining from physicochemical properties.
#   In Decision Support Systems>, Elsevier, 47(4):547-553. ISSN: 0167-9236.

#   In the above reference, two datasets were created, using red and white wine samples.
#   The inputs include objective tests (e.g. PH values) and the output is based on sensory data
#   (median of at least 3 evaluations made by wine experts). Each expert graded the wine quality 
#   between 0 (very bad) and 10 (very excellent). Several data mining methods were applied to model
#   these datasets under a regression approach. The support vector machine model achieved the
#   best results. Several metrics were computed: MAD, confusion matrix for a fixed error tolerance (T),
#   etc. Also, we plot the relative importances of the input variables (as measured by a sensitivity
#   analysis procedure).
 
# 4. Relevant Information:

#    These datasets can be viewed as classification or regression tasks.
#    The classes are ordered and not balanced (e.g. there are munch more normal wines than
#    excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent
#    or poor wines. Also, we are not sure if all input variables are relevant. So
#    it could be interesting to test feature selection methods. 

# 5. Number of Instances: red wine - 1599; white wine - 4898. 

# 6. Number of Attributes: 11 + output attribute
  
#    Note: several of the attributes may be correlated, thus it makes sense to apply some sort of
#    feature selection.

# 7. Attribute information:

#    For more information, read [Cortez et al., 2009].

#    Input variables (based on physicochemical tests):
#    1 - fixed acidity
#    2 - volatile acidity
#    3 - citric acid
#    4 - residual sugar
#    5 - chlorides
#    6 - free sulfur dioxide
#    7 - total sulfur dioxide
#    8 - density
#    9 - pH
#    10 - sulphates
#    11 - alcohol
#    Output variable (based on sensory data): 
#    12 - quality (score between 0 and 10)

# 8. Missing Attribute Values: None
# In[ ]:


import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pickle

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os


# In[ ]:


dr = pd.read_csv('data/original/wine-quality/winequality-red.csv', delimiter=';')
dr['color'] = 0
dw = pd.read_csv('data/original/wine-quality/winequality-white.csv', delimiter=';')
dw['color'] = 1

outdir = 'data/wine-quality3'
os.makedirs(outdir, exist_ok=True)


# In[ ]:


dat = dr.append(dw)


# In[ ]:


def create_encoder_dict(dataset, cate_cols, nume_cols):
    enc_dict = dict()

    # Create encoder by using train data
    print('categorical')
    for k in tqdm(cate_cols):
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        enc = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, [0])
            ]
        )

        enc.fit(dataset[k].astype(str).values.reshape(-1, 1))
        enc_dict[k] = enc

    print('numerical')
    for k in tqdm(nume_cols):
        scaler = MinMaxScaler()
        scaler.fit(dataset[k].values.reshape(-1, 1))
        enc_dict[k] = scaler

    return enc_dict


# In[ ]:


def encode(dataset, cate_cols, nume_cols, enc_dict):
    #     dataset2 = dataset.copy()

    print('categorical')
    for k in tqdm(cate_cols):
        print(k)
        out = enc_dict[k].transform(dataset[k].astype(str).values.reshape(-1, 1))
        if not type(out) == np.ndarray:
            out = out.todense()
        keys = [f'{k}_{i}' for i in range(out.shape[1])]
        dataset[keys] = out
        dataset = dataset.drop(k, axis=1)

    print('numerical')
    for k in tqdm(nume_cols):
        print(k)
        out = enc_dict[k].transform(dataset[k].values.reshape(-1, 1)).flatten()
        dataset[k] = out

    return dataset


# In[ ]:


for i in range(len(dat.columns)):
    if (dat.dtypes[i] == 'object') and dat.columns[i] != 'class':
        print('--')
        print(i)
        col = dat.columns[i]
        print(dat.columns[i])
        print(dat.dtypes[i])
        pic = dat[[col]][dat[col].apply(lambda s:pd.to_numeric(s, errors='coerce')).isnull()]
        print(pic)
        dat[col] = dat[col].astype('float')
        dat[col] = dat[col].fillna(dat[col].mean())


# In[ ]:


h = dat.columns
dat.columns = [v.lower().replace(' ', '_') for v in h]


# In[ ]:


h = dat.columns
print(h)


# In[ ]:


nume_cols = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
       'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
       'ph', 'sulphates', 'alcohol', 'color']
cate_cols = []


# In[ ]:


enc_dict = create_encoder_dict(dat, cate_cols, nume_cols)
dat = encode(dat, cate_cols, nume_cols, enc_dict)


# In[ ]:


dat['user_id'] = range(0,len(dat))
dat = dat.set_index('user_id')


# In[ ]:


dat['quality'] = dat['quality'] >= 7


# In[ ]:


dat['quality'] = dat['quality'].astype(int)


# In[ ]:


tr_y = dat[['quality']].astype(int)
tr_x = dat.drop('quality', axis=1)


# In[ ]:



tr_x, va_x, tr_y, va_y = train_test_split(tr_x, tr_y, test_size = 0.2)
va_x, te_x, va_y, te_y = train_test_split(va_x, va_y, test_size = 0.5)


# In[ ]:


columns = []
columns.append([
    'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar'
])
columns.append([
    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density'
])
columns.append([
    'ph', 'sulphates', 'alcohol', 'color', 
])


# In[ ]:


def create_dataset3(dx, dy, columns, header = 'tr'):
    x_list = []
    y_list = []
    indices_list = []
    f_col_list = []
    l_col_list = []
    for cid in range(len(columns)):
        x_list.append(dx[columns[cid]])
        y_list.append(dy)
        indices_list.append(dx[columns[cid]].index)
        f_col_list.append(columns[cid]) #tr_f_col = columns[cid]
        l_col_list.append(dy.columns) #tr_l_col = ''
    

    pickle.dump(x_list, open(f'{outdir}/{header}_feature_v3.pkl', 'wb'))
    pickle.dump(y_list, open(f'{outdir}/{header}_label_v3.pkl', 'wb'))
    pickle.dump(indices_list, open(f'{outdir}/{header}_indices_v3.pkl', 'wb'))
    pickle.dump(f_col_list, open(f'{outdir}/{header}_f_col_v3.pkl', 'wb'))
    pickle.dump(l_col_list, open(f'{outdir}/{header}_l_col_v3.pkl', 'wb'))


# In[ ]:


create_dataset3(tr_x, tr_y, columns, header = 'tr')
create_dataset3(va_x, va_y, columns, header = 'va')
create_dataset3(te_x, te_y, columns, header = 'te')

