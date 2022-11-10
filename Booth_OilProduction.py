# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 15:26:47 2021

@author: booth-s
"""

import pandas as pd
import numpy  as np
import statsmodels.api as sm
import statsmodels.tools.eval_measures as em
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from AdvancedAnalytics.ReplaceImputeEncode import ReplaceImputeEncode, DT
from AdvancedAnalytics.Regression import linreg, stepwise
from sklearn import linear_model

# Reading in the data frame from the excel file
df = pd.read_excel("OilProduction.xlsx")


# Creating the attribute map for the data
# Have to make sure that Operator and County as a tuple because otherwise you eliminate all the variables
attribute_map = {  
    'Log_Cum_Production': [DT.Interval, (8, 15)], 
    'Log_Proppant_LB': [DT.Interval, (6, 18)], 
    'Log_Carbonate': [DT.Interval, (-4, 4)], 
    'Log_Frac_Fluid_GL': [DT.Interval, (7, 18)], 
    'Log_GrossPerforatedInterval': [DT.Interval, (4, 9)], 
    'Log_LowerPerforation_xy': [DT.Interval, (8, 10)], 
    'Log_UpperPerforation_xy': [DT.Interval, (8, 10)], 
    'Log_TotalDepth': [DT.Interval, (8, 10)], 
    'N_Stages': [DT.Interval, (2, 14)], 
    'X_Well': [DT.Interval, (-100, -95)],
    'Y_Well': [DT.Interval, (30, 35)],
    'Operator': [DT.Nominal, tuple(range(1, 28))],
    'County': [DT.Nominal, tuple(range(1, 14))]
    }


# Defining the target as Log_Cum_Production, in the in class example another data frame is created because there are 
# two targets in on df and you don't want to use the second target in the other regression so below is the appropriate
# way to set your target, identify where you have missing values in each column, then replace those missing values
# with the mean (if an interval variable) or mode (if nominal).

# Important to note that the target actually has missing #s, but you CANNOT impute a target, hence setting it to no impute
target = 'Log_Cum_Production'
rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot',
        no_impute=[target], interval_scale = None, drop=True, display=True)
encoded_df = rie.fit_transform(df).dropna()

# This is where you run the actual stepwise to chose your variables. Important to note here is target was defined
# in line 42
sw = stepwise(encoded_df, target, reg="linear", method="stepwise", 
              crit_in=0.1, crit_out=0.1, verbose=True)
print("\nRunning Stepwise")
selected = sw.fit_transform()

# Now this is running the actual model using statsmodels and not sklearn, need to clarify what exactly the difference
# is between the two packages. Clearly with statsmodels you have to specify for the constant.
y = encoded_df[target]
y = np.ravel(y)
X  = encoded_df[selected]
Xc = sm.add_constant(X)

# This is splitting the dataset as we defined it in lines 58 - 61 into training and validation with a regular
# stepwise model. 
X_train, X_validate, y_train, y_validate = \
    train_test_split(Xc, y, test_size =0.3, random_state = 12345)
    
model = sm.OLS(y_train, X_train)
results = model.fit()

print("\n*******************************************************")
print("                                   Target: Log_Cum_Production        ")
print(results.summary())
print("\n*******************************************************")

# Compares the Training and Validate models done with stepwise regression
print("Advanced Analytics Display Split Metrics:")
linreg.display_split_metrics(results, X_train, y_train, X_validate, y_validate)
    
# Do the LASSO Regression with sklearn package, remember data is already split in steps 66 - 67
clf = linear_model.Lasso()
results = clf.fit(X_train, y_train)

# Compares the results of the training vs validation model for LASSO
linreg.display_split_metrics(results, X_train, y_train, X_validate, y_validate)

# Displays the overall model for the training set of LASSO
linreg.display_coef(results, X_train, y_train)
linreg.display_metrics(results, X_train, y_train)

#Displays the overall model for the validation set of LASSO
linreg.display_coef(results, X_validate, y_validate)
linreg.display_metrics(results, X_validate, y_validate)