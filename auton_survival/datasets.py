# coding=utf-8
# MIT License

# Copyright (c) 2020 Carnegie Mellon University, Auton Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Utility functions to load standard datasets to train and evaluate the
Deep Survival Machines models.
"""


import io
import pkgutil

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import torchvision

def increase_censoring(e, t, p, random_seed=0):

  np.random.seed(random_seed)

  uncens = np.where(e == 1)[0]
  mask = np.random.choice([False, True], len(uncens), p=[1-p, p])
  toswitch = uncens[mask]

  e[toswitch] = 0
  t_ = t[toswitch]

  newt = []
  for t__ in t_:
    newt.append(np.random.uniform(1, t__))
  t[toswitch] = newt

  return e, t

def _load_framingham_dataset(brut=False):
  """Helper function to load and preprocess the Framingham dataset.
  The Framingham Dataset is a subset of 4,434 participants of the well known,
  ongoing Framingham Heart study [1] for studying epidemiology for
  hypertensive and arteriosclerotic cardiovascular disease. It is a popular
  dataset for longitudinal survival analysis with time dependent covariates.
  Parameters
  ----------
  sequential: bool
    If True returns a list of np.arrays for each individual.
    else, returns collapsed results for each time step. To train
    recurrent neural models you would typically use True.
  References
  ----------
  [1] Dawber, Thomas R., Gilcin F. Meadors, and Felix E. Moore Jr.
  "Epidemiological approaches to heart disease: the Framingham Study."
  American Journal of Public Health and the Nations Health 41.3 (1951).
  """

  data = pkgutil.get_data(__name__, 'datasets/framingham.csv')
  data = pd.read_csv(io.BytesIO(data))

  data = pd.DataFrame(data, columns=['RANDID', 'SEX', 'TOTCHOL', 'AGE', 'SYSBP', 'DIABP', 'CURSMOKE','CIGPDAY', 'BMI', 
                                           'DIABETES', 'BPMEDS', 'HEARTRTE', 'GLUCOSE', 'educ','PREVCHD', 'PREVAP', 'PREVMI', 
                                           'PREVSTRK', 'PREVHYP', 'TIME', 'PERIOD','HDLC', 'LDLC', 'DEATH', 'ANGINA', 'HOSPMI', 
                                           'MI_FCHD', 'ANYCHD','STROKE', 'CVD', 'HYPERTEN', 'TIMEAP', 'TIMEMI', 'TIMEMIFC', 'TIMECHD',
                                           'TIMESTRK', 'TIMECVD', 'TIMEDTH', 'TIMEHYP'])
  time = (data['TIMEDTH'] - data['TIME']).values
  data['time'] = time
  event = data['DEATH'].values
  data['event'] = event

  data = data.drop(columns=['RANDID', 'TIME', 'PERIOD', 'DEATH', 'TIMEDTH','TIMEMI', 'TIMEMIFC', 'TIMECHD', 'TIMESTRK', 'TIMECVD', 'TIMEHYP', 'TIMEAP'])
  time = pd.DataFrame(time, columns=['time'])
  event = pd.DataFrame(event, columns=['event'])


  dat_cat = data[['SEX', 'CURSMOKE', 'DIABETES', 'BPMEDS','educ', 'PREVCHD', 'PREVAP', 'PREVMI',
                  'PREVSTRK', 'PREVHYP', 'ANGINA', 'HOSPMI','MI_FCHD', 'ANYCHD', 'STROKE', 'CVD', 'HYPERTEN']]
  dat_num = data[['TOTCHOL', 'AGE', 'SYSBP', 'DIABP','CIGPDAY', 'BMI', 'HEARTRTE', 'GLUCOSE', 'LDLC', 'HDLC']]

  #get dummies pandas for categorical data
  dat_cat = pd.get_dummies(dat_cat, drop_first=True)

  #scale the data
  dat_num = StandardScaler().fit_transform(dat_num)

  dat_num = pd.DataFrame(dat_num, columns=['TOTCHOL', 'AGE', 'SYSBP', 'DIABP','CIGPDAY', 'BMI', 'HEARTRTE', 'GLUCOSE', 'LDLC', 'HDLC'])

  data = pd.concat([dat_cat, dat_num], axis=1)

  data = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(data)

  if brut :
    return data, time, event

  data = pd.DataFrame(data, columns=['SEX', 'CURSMOKE', 'DIABETES', 'BPMEDS', 'educ', 'PREVCHD', 'PREVAP',
       'PREVMI', 'PREVSTRK', 'PREVHYP', 'ANGINA', 'HOSPMI', 'MI_FCHD',
       'ANYCHD', 'STROKE', 'CVD', 'HYPERTEN', 'TOTCHOL', 'AGE', 'SYSBP',
       'DIABP', 'CIGPDAY', 'BMI', 'HEARTRTE', 'GLUCOSE', 'LDLC', 'HDLC'])
  
  data = pd.concat([data, time, event], axis=1)

  return data, time, event

def _load_pbc_dataset(brut=False):
  """Helper function to load and preprocess the PBC dataset
  The Primary biliary cirrhosis (PBC) Dataset [1] is well known
  dataset for evaluating survival analysis models with time
  dependent covariates.
  Parameters
  ----------
  sequential: bool
    If True returns a list of np.arrays for each individual.
    else, returns collapsed results for each time step. To train
    recurrent neural models you would typically use True.
  References
  ----------
  [1] Fleming, Thomas R., and David P. Harrington. Counting processes and
  survival analysis. Vol. 169. John Wiley & Sons, 2011.
  """

  data = pkgutil.get_data(__name__, 'datasets/pbc2.csv')
  data = pd.read_csv(io.BytesIO(data))

  time = (data['years'] - data['year']).values
  event = data['status2'].values
  data['histologic'] = data['histologic'].astype(str)

  #convert and concat features, time and event numpy array
  data = pd.DataFrame(data, columns=['sno.', 'id', 'years', 'status', 'drug', 'age', 'sex', 'year',
       'ascites', 'hepatomegaly', 'spiders', 'edema', 'serBilir', 'serChol', 'albumin', 'alkaline', 'SGOT', 'platelets', 
       'prothrombin', 'histologic'])
  time = pd.DataFrame(time, columns=['time'])
  event = pd.DataFrame(event, columns=['event'])

  age = data['age'] + data['years']
  data['age'] = age
  data = data.drop(columns=['years','id','status','year','sno.'])

  dat_cat = data[['drug', 'sex', 'ascites', 'hepatomegaly',
                  'spiders', 'edema', 'histologic']]
  dat_num = data[['age','serBilir', 'serChol', 'albumin', 'alkaline',
                  'SGOT', 'platelets', 'prothrombin']]

  #get dummies pandas for categorical data and boolean type
  dat_cat = pd.get_dummies(dat_cat, drop_first=True)

  dat_num = StandardScaler().fit_transform(dat_num)

  dat_num = pd.DataFrame(dat_num, columns=['age','serBilir', 'serChol', 'albumin', 'alkaline',
                  'SGOT', 'platelets', 'prothrombin'])

  data = pd.concat([dat_cat, dat_num], axis=1)
  
  #replace nan values with mean
  data = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(data)

  if brut :
    return data, time, event

  #transform the data in dataframe
  data = pd.DataFrame(data, columns=['drug_placebo', 'sex_male', 'ascites_Yes', 'hepatomegaly_Yes',
       'spiders_Yes', 'edema_edema despite diuretics',
       'edema_edema no diuretics', 'histologic_2', 'histologic_3',
       'histologic_4', 'age', 'serBilir', 'serChol', 'albumin', 'alkaline',
       'SGOT', 'platelets', 'prothrombin'])
  
  data['drug_placebo'] = data['drug_placebo'].astype(int)
  data['sex_male'] = data['sex_male'].astype(int)
  data['ascites_Yes'] = data['ascites_Yes'].astype(int)
  data['hepatomegaly_Yes'] = data['hepatomegaly_Yes'].astype(int)
  data['spiders_Yes'] = data['spiders_Yes'].astype(int)
  data['edema_edema despite diuretics'] = data['edema_edema despite diuretics'].astype(int)
  data['edema_edema no diuretics'] = data['edema_edema no diuretics'].astype(int)
  data['histologic_2'] = data['histologic_2'].astype(int)
  data['histologic_3'] = data['histologic_3'].astype(int)
  data['histologic_4'] = data['histologic_4'].astype(int)

  data = pd.concat([data, time, event], axis=1)

  return data, time, event

def load_support():

  """Helper function to load and preprocess the SUPPORT dataset.
  The SUPPORT Dataset comes from the Vanderbilt University study
  to estimate survival for seriously ill hospitalized adults [1].
  Please refer to http://biostat.mc.vanderbilt.edu/wiki/Main/SupportDesc.
  for the original datasource.

  References
  ----------
  [1]: Knaus WA, Harrell FE, Lynn J et al. (1995): The SUPPORT prognostic
  model: Objective estimates of survival for seriously ill hospitalized
  adults. Annals of Internal Medicine 122:191-203.
  """

  data = pkgutil.get_data(__name__, 'datasets/support2.csv')
  data = pd.read_csv(io.BytesIO(data))

  drop_cols = ['death', 'd.time']

  outcomes = data.copy()
  outcomes['event'] =  data['death']
  outcomes['time'] = data['d.time']
  outcomes = outcomes[['event', 'time']]

  cat_feats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
  num_feats = ['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp',
              'temp', 'pafi', 'alb', 'bili', 'crea', 'sod', 'ph',
              'glucose', 'bun', 'urine', 'adlp', 'adls']

  return outcomes, data[cat_feats + num_feats]




# def _load_support_dataset():
#   """Helper function to load and preprocess the SUPPORT dataset.
#   The SUPPORT Dataset comes from the Vanderbilt University study
#   to estimate survival for seriously ill hospitalized adults [1].
#   Please refer to http://biostat.mc.vanderbilt.edu/wiki/Main/SupportDesc.
#   for the original datasource.
#   References
#   ----------
#   [1]: Knaus WA, Harrell FE, Lynn J et al. (1995): The SUPPORT prognostic
#   model: Objective estimates of survival for seriously ill hospitalized
#   adults. Annals of Internal Medicine 122:191-203.
#   """

#   data = pkgutil.get_data(__name__, 'datasets/support2.csv')
#   data = pd.read_csv(io.BytesIO(data))
#   x1 = data[['age', 'num.co', 'meanbp', 'wblc', 'hrt', 'resp', 'temp',
#              'pafi', 'alb', 'bili', 'crea', 'sod', 'ph', 'glucose', 'bun',
#              'urine', 'adlp', 'adls']]

#   catfeats = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca']
#   x2 = pd.get_dummies(data[catfeats])

#   x = np.concatenate([x1, x2], axis=1)
#   t = data['d.time'].values
#   e = data['death'].values

#   x = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(x)
#   x = StandardScaler().fit_transform(x)

#   remove = ~np.isnan(t)

#   return x[remove], t[remove], e[remove]

def _load_mnist():
  """Helper function to load and preprocess the MNIST dataset.
  The MNIST database of handwritten digits, available from this page, has a
  training set of 60,000 examples, and a test set of 10,000 examples.
  It is a good database for people who want to try learning techniques and
  pattern recognition methods on real-world data while spending minimal
  efforts on preprocessing and formatting [1].
  Please refer to http://yann.lecun.com/exdb/mnist/.
  for the original datasource.
  References
  ----------
  [1]: LeCun, Y. (1998). The MNIST database of handwritten digits.
  http://yann.lecun.com/exdb/mnist/.
  """

  train = torchvision.datasets.MNIST(root='datasets/',
                                     train=True, download=True)
  x = train.data.numpy()
  x = np.expand_dims(x, 1).astype(float)
  t = train.targets.numpy().astype(float) + 1

  e, t = increase_censoring(np.ones(t.shape), t, p=.5)

  return x, t, e

def load_synthetic_cf_phenotyping():

  data = pkgutil.get_data(__name__, 'datasets/synthetic_dataset.csv')
  data = pd.read_csv(io.BytesIO(data))

  outcomes = data[['event', 'time', 'uncensored time treated',
                   'uncensored time control', 'Z','Zeta']]

  features = data[['X1','X2','X3','X4','X5','X6','X7','X8']]
  interventions = data['intervention']

  return outcomes, features, interventions

def load_dataset(dataset='SUPPORT',brut=False):
  """Helper function to load datasets to test Survival Analysis models.
  Currently implemented datasets include:\n
  **SUPPORT**: This dataset comes from the Vanderbilt University study
  to estimate survival for seriously ill hospitalized adults [1].
  (Refer to http://biostat.mc.vanderbilt.edu/wiki/Main/SupportDesc.
  for the original datasource.)\n
  **PBC**: The Primary biliary cirrhosis dataset [2] is well known
  dataset for evaluating survival analysis models with time
  dependent covariates.\n
  **FRAMINGHAM**: This dataset is a subset of 4,434 participants of the well
  known, ongoing Framingham Heart study [3] for studying epidemiology for
  hypertensive and arteriosclerotic cardiovascular disease. It is a popular
  dataset for longitudinal survival analysis with time dependent covariates.\n
  **SYNTHETIC**: This is a non-linear censored dataset for counterfactual
  time-to-event phenotyping. Introduced in [4], the dataset is generated
  such that the treatment effect is heterogenous conditioned on the covariates.

  References
  -----------
  [1]: Knaus WA, Harrell FE, Lynn J et al. (1995): The SUPPORT prognostic
  model: Objective estimates of survival for seriously ill hospitalized
  adults. Annals of Internal Medicine 122:191-203.\n
  [2] Fleming, Thomas R., and David P. Harrington. Counting processes and
  survival analysis. Vol. 169. John Wiley & Sons, 2011.\n
  [3] Dawber, Thomas R., Gilcin F. Meadors, and Felix E. Moore Jr.
  "Epidemiological approaches to heart disease: the Framingham Study."
  American Journal of Public Health and the Nations Health 41.3 (1951).\n
  [4] Nagpal, C., Goswami M., Dufendach K., and Artur Dubrawski.
  "Counterfactual phenotyping for censored Time-to-Events" (2022).

  Parameters
  ----------
  dataset: str
      The choice of dataset to load. Currently implemented is 'SUPPORT',
      'PBC' and 'FRAMINGHAM'.
  **kwargs: dict
      Dataset specific keyword arguments.

  Returns
  ----------
  tuple: (np.ndarray, np.ndarray, np.ndarray)
      A tuple of the form of \( (x, t, e) \) where \( x \)
      are the input covariates, \( t \) the event times and
      \( e \) the censoring indicators.
  """

  if dataset == 'SUPPORT':
    return load_support()
  if dataset == 'PBC':
    return _load_pbc_dataset(brut)
  if dataset == 'FRAMINGHAM':
    return _load_framingham_dataset(brut)
  if dataset == 'MNIST':
    return _load_mnist()
  if dataset == 'SYNTHETIC':
    return load_synthetic_cf_phenotyping()
  else:
    raise NotImplementedError('Dataset '+dataset+' not implemented.')
