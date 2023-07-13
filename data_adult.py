#!/usr/bin/env python
# coding: utf-8

# # Adult Dataset

# 1. Download adult.zip from https://archive.ics.uci.edu/dataset/2/adult
# 2. Save the zip as ```data/original/adult/adult.zip```
# 3. ```cd data/original/adult```
# 4. ```unzip adult.zip```



# | This data was extracted from the census bureau database found at
# | http://www.census.gov/ftp/pub/DES/www/welcome.html
# | Donor: Ronny Kohavi and Barry Becker,
# |        Data Mining and Visualization
# |        Silicon Graphics.
# |        e-mail: ronnyk@sgi.com for questions.
# | Split into train-test using MLC++ GenCVFiles (2/3, 1/3 random).
# | 48842 instances, mix of continuous and discrete    (train=32561, test=16281)
# | 45222 if instances with unknown values are removed (train=30162, test=15060)
# | Duplicate or conflicting instances : 6
# | Class probabilities for adult.all file
# | Probability for the label '>50K'  : 23.93% / 24.78% (without unknowns)
# | Probability for the label '<=50K' : 76.07% / 75.22% (without unknowns)
# |
# | Extraction was done by Barry Becker from the 1994 Census database.  A set of
# |   reasonably clean records was extracted using the following conditions:
# |   ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
# |
# | Prediction task is to determine whether a person makes over 50K
# | a year.
# |
# | First cited in:
# | @inproceedings{kohavi-nbtree,
# |    author={Ron Kohavi},
# |    title={Scaling Up the Accuracy of Naive-Bayes Classifiers: a
# |           Decision-Tree Hybrid},
# |    booktitle={Proceedings of the Second International Conference on
# |               Knowledge Discovery and Data Mining},
# |    year = 1996,
# |    pages={to appear}}
# |
# | Error Accuracy reported as follows, after removal of unknowns from
# |    train/test sets):
# |    C4.5       : 84.46+-0.30
# |    Naive-Bayes: 83.88+-0.30
# |    NBTree     : 85.90+-0.28
# |
# |
# | Following algorithms were later run with the following error rates,
# |    all after removal of unknowns and using the original train/test split.
# |    All these numbers are straight runs using MLC++ with default values.
# |
# |    Algorithm               Error
# | -- ----------------        -----
# | 1  C4.5                    15.54
# | 2  C4.5-auto               14.46
# | 3  C4.5 rules              14.94
# | 4  Voted ID3 (0.6)         15.64
# | 5  Voted ID3 (0.8)         16.47
# | 6  T2                      16.84
# | 7  1R                      19.54
# | 8  NBTree                  14.10
# | 9  CN2                     16.00
# | 10 HOODG                   14.82
# | 11 FSS Naive Bayes         14.05
# | 12 IDTM (Decision table)   14.46
# | 13 Naive-Bayes             16.12
# | 14 Nearest-neighbor (1)    21.42
# | 15 Nearest-neighbor (3)    20.35
# | 16 OC1                     15.04
# | 17 Pebls                   Crashed.  Unknown why (bounds WERE increased)
# |
# | Conversion of original data as follows:
# | 1. Discretized agrossincome into two ranges with threshold 50,000.
# | 2. Convert U.S. to US to avoid periods.
# | 3. Convert Unknown to "?"
# | 4. Run MLC++ GenCVFiles to generate data,test.
# |
# | Description of fnlwgt (final weight)
# |
# | The weights on the CPS files are controlled to independent estimates of the
# | civilian noninstitutional population of the US.  These are prepared monthly
# | for us by Population Division here at the Census Bureau.  We use 3 sets of
# | controls.
# |  These are:
# |          1.  A single cell estimate of the population 16+ for each state.
# |          2.  Controls for Hispanic Origin by age and sex.
# |          3.  Controls by Race, age and sex.
# |
# | We use all three sets of controls in our weighting program and "rake" through
# | them 6 times so that by the end we come back to all the controls we used.
# |
# | The term estimate refers to population totals derived from CPS by creating
# | "weighted tallies" of any specified socio-economic characteristics of the
# | population.
# |
# | People with similar demographic characteristics should have
# | similar weights.  There is one important caveat to remember
# | about this statement.  That is that since the CPS sample is
# | actually a collection of 51 state samples, each with its own
# | probability of selection, the statement only applies within
# | state.


# >50K, <=50K.

# age: continuous.
# workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# fnlwgt: continuous.
# education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# education-num: continuous.
# marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# sex: Female, Male.
# capital-gain: continuous.
# capital-loss: continuous.
# hours-per-week: continuous.
# native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
# In[ ]:


import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle
import os


# In[ ]:


h = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education_num',
    'marital_status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'native_country',
    'over_50k'
]
nume_cate = [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1]
nume_cols = [h[i] for i in range(len(nume_cate)) if nume_cate[i] == 0]
cate_cols = [h[i] for i in range(len(nume_cate)) if nume_cate[i] == 1]


# In[ ]:


dat = pd.read_csv('data/original/adult/adult.data', header=None)
te = pd.read_csv('data/original/adult/adult.test', header=None, skiprows=1)
outdir = 'data/adult3'
os.makedirs(outdir, exist_ok=True)


# In[ ]:


dat.columns = h


# In[ ]:


te.columns = h


# In[ ]:


te.over_50k=te.over_50k.str.split('.', expand=True)[0]


# In[ ]:





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


enc_dict = create_encoder_dict(dat, cate_cols, nume_cols)


# In[ ]:


dat = encode(dat, cate_cols, nume_cols, enc_dict)
te = encode(te, cate_cols, nume_cols, enc_dict)


# In[ ]:


dat['user_id'] = range(0,len(dat))
dat = dat.set_index('user_id')
te['user_id'] = range(len(dat), len(dat)+len(te))
te = te.set_index('user_id')


# In[ ]:


tr_y = dat[['over_50k_1']].astype(int)
tr_x = dat.drop(['over_50k_0', 'over_50k_1'], axis=1)


# In[ ]:




tr_x, va_x, tr_y, va_y = train_test_split(tr_x, tr_y, test_size = 0.1)


# In[ ]:


te_y = te[['over_50k_1']].astype(int)
te_x = te.drop(['over_50k_0', 'over_50k_1'], axis=1)


# In[ ]:


print(tr_x.shape,tr_y.shape)
print(va_x.shape,va_y.shape)
print(te_x.shape,te_y.shape)


# In[ ]:





# In[ ]:


columns = []
columns.append([
    'age',
    'workclass_0', 'workclass_1', 'workclass_2',
    'workclass_3', 'workclass_4', 'workclass_5', 'workclass_6', 'workclass_7', 'workclass_8', 
    'fnlwgt',
    'education_0', 'education_1',
    'education_2', 'education_3', 'education_4', 'education_5',
    'education_6', 'education_7', 'education_8', 'education_9',
    'education_10', 'education_11', 'education_12', 'education_13',
    'education_14', 'education_15', 
    'education_num',

])

columns.append([
    'marital_status_0', 'marital_status_1',
    'marital_status_2', 'marital_status_3', 'marital_status_4',
    'marital_status_5', 'marital_status_6', 
    'occupation_0', 'occupation_1',
    'occupation_2', 'occupation_3', 'occupation_4', 'occupation_5',
    'occupation_6', 'occupation_7', 'occupation_8', 'occupation_9',
    'occupation_10', 'occupation_11', 'occupation_12', 'occupation_13',
    'occupation_14',    'relationship_0', 'relationship_1', 'relationship_2',
    'relationship_3', 'relationship_4', 'relationship_5', 
    'race_0', 'race_1', 'race_2', 'race_3', 'race_4', 
    'sex_0', 'sex_1',

])


columns.append([
    'capital_gain',
    'capital_loss',
    'hours_per_week',
    'native_country_0', 'native_country_1', 'native_country_2',
    'native_country_3', 'native_country_4', 'native_country_5',
    'native_country_6', 'native_country_7', 'native_country_8',
    'native_country_9', 'native_country_10', 'native_country_11',
    'native_country_12', 'native_country_13', 'native_country_14',
    'native_country_15', 'native_country_16', 'native_country_17',
    'native_country_18', 'native_country_19', 'native_country_20',
    'native_country_21', 'native_country_22', 'native_country_23',
    'native_country_24', 'native_country_25', 'native_country_26',
    'native_country_27', 'native_country_28', 'native_country_29',
    'native_country_30', 'native_country_31', 'native_country_32',
    'native_country_33', 'native_country_34', 'native_country_35',
    'native_country_36', 'native_country_37', 'native_country_38',
    'native_country_39', 'native_country_40', 'native_country_41',
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

