# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 03:59:17 2021

@author: Ziba
"""

# -- RawData_Manipulation_Phase 1 --

# importing pandas as pd
import pandas as pd
  
# Creating the dataframe 
df = pd.read_csv("rawdata.csv")
  
# Print the dataframe
df


# finding sum over index axis
# By default the axis is set to 0
df.sum(axis = 0, skipna = True)

# importing pandas as pd
import pandas as pd
  
# Creating the dataframe 
df = pd.read_csv("rawdata.csv")
  
# sum over the column axis.
df.sum(axis = 1, skipna = True)

# Print the DataFrame
print(df)

print(df)

import numpy as np
conditions = [
        (df['Quality Index'] <= 36.66) & (df['Quality Index'] > 26.09),
        (df['Quality Index'] <= 26.09),
        (df['Quality Index'] <= 13.41),
                ]

# create a list of the values we want to assign for each condition
values = ['Good', 'QuiteGood', 'Bad']

# create a new column and use np.select to assign values to it using our lists as arguments
df['Fish Condition'] = np.select(conditions, values)

# display updated DataFrame
df.head() 

print(df)
df