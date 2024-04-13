#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import datetime as dt
import zipfile
from io import BytesIO, StringIO
from nselib.libutil import *
from nselib.constants import *
from nselib import capital_market
import nselib


# In[2]:


#Enter name of the script to be run

stock_name = input('Enter script name for the stock: ')
df = capital_market.price_volume_and_deliverable_position_data(stock_name, period='1M')


# In[3]:


#Consider only Equities related data
df = df[df['Series'] == 'EQ']
df.tail(13)


# # In the above section check if there are any blanks or missing values. If so, use the manual report for extracting this data

# In[4]:


# Remove commas and convert to integer
df['No.ofTrades'] = df['No.ofTrades'].str.replace(',', '').astype(int)
df['TotalTradedQuantity'] = df['TotalTradedQuantity'].str.replace(',', '').astype(int)
df['DeliverableQty'] = df['DeliverableQty'].str.replace(',', '').astype('float64')
df['TurnoverInRs'] = df['TurnoverInRs'].str.replace(',', '').astype('float64')
print(df.dtypes)


# In[5]:


# Calculate TTQ/NT
df['TTQ/NT'] = df['TotalTradedQuantity'] / df['No.ofTrades']
df


# In[6]:


# Convert the DataFrame to pandas DataFrame
df = pd.DataFrame(df)


# In[7]:


# List of columns to be converted to float
columns_to_convert = ['AveragePrice','TotalTradedQuantity','TurnoverInRs','No.ofTrades','DeliverableQty','%DlyQttoTradedQty']


# In[8]:


# Remove commas from numeric columns and convert to float
for col in columns_to_convert:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col].replace('-', np.nan).str.replace(',', ''), errors='coerce')


# In[9]:


df.tail(13)


# In[10]:


# Fill NaN values with previous column value
df = df.fillna(method='ffill', axis=0)

df.tail()


# In[11]:


# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Sort dataframe by date
df = df.sort_values(by='Date')

# Select required columns
selected_columns = ['AveragePrice', 'TotalTradedQuantity', 'TurnoverInRs', 'No.ofTrades', 'DeliverableQty', 'TTQ/NT']
selected_df = df[selected_columns]

# Calculate rolling means for the specified days
rolling_means = pd.DataFrame()
rolling_means['Last 3 Days'] = selected_df.rolling(window=3).mean().iloc[-1]
rolling_means['Last 5 Days'] = selected_df.rolling(window=5).mean().iloc[-1]
rolling_means['Last 8 Days'] = selected_df.rolling(window=8).mean().iloc[-1]
rolling_means['Last 13 Days'] = selected_df.rolling(window=13).mean().iloc[-1]
rolling_means


# In[12]:


# Create a DataFrame
df_t = pd.DataFrame(rolling_means)

# Transpose the DataFrame and add a heading to the first column
rolling_means = df_t.T.reset_index()
rolling_means.columns = ['Period', 'AveragePrice', 'TotalTradedQuantity','TurnoverInRs','No.ofTrades','DeliverableQty','TTQ/NT']
rolling_means


# In[13]:


# Convert the DataFrame to integer type, specifying int64 for Turnover
rolling_means_filled_whole = rolling_means.astype({
    'AveragePrice': int, 
    'TotalTradedQuantity': int, 
    'TurnoverInRs': np.int64, 
    'No.ofTrades': int, 
    'DeliverableQty': int, 
})

df1 = rolling_means_filled_whole
df1


# In[14]:


# Calculate the sums for the last 3 days, 5 days, 8 days and 13 days for calculating percentile of delivered quantity to traded quantity
df2 = pd.DataFrame({
    'Last 3 Days': [
        df['DeliverableQty'].iloc[-3:].sum(),
        df['TotalTradedQuantity'].iloc[-3:].sum()
    ],
    'Last 5 Days': [
        df['DeliverableQty'].iloc[-5:].sum(),
        df['TotalTradedQuantity'].iloc[-5:].sum()
    ],
    'Last 8 Days': [
        df['DeliverableQty'].iloc[-8:].sum(),
        df['TotalTradedQuantity'].iloc[-8:].sum()
    ],
    'Last 13 Days': [
        df['DeliverableQty'].iloc[-13:].sum(),
        df['TotalTradedQuantity'].iloc[-13:].sum()
    ]
}, index=['Sum of Deliverable Qty', 'Sum of Total Traded Quantity'])

# Display the new dataframe 'df2'
df2


# In[15]:


# Create a DataFrame
df_t = pd.DataFrame(df2)

# Transpose the DataFrame and add a heading to the first column
df2 = df_t.T.reset_index()
df2.columns = ['Period', 'Sum of Deliverable Qty', 'Sum of Total Traded Quantity']
df2


# In[16]:


# Calculate the percentage of deliverable quantity to total traded quantity
df2['Per_DelQty/TTQ'] = df2['Sum of Deliverable Qty'] / df2['Sum of Total Traded Quantity'] * 100
df2


# In[17]:


df2 = df2.drop(['Sum of Deliverable Qty', 'Sum of Total Traded Quantity'], axis=1)
df2


# In[18]:


df1


# In[19]:


print(df1.columns)


# In[20]:


print(df2.columns)


# In[21]:


# Reset index for df1 and df2
df1_reset = df1.reset_index(drop=True)
df2_reset = df2.reset_index(drop=True)

# Merge df1_reset and df2_reset based on the index
df_merged = pd.merge(df1_reset, df2_reset, left_index=True, right_index=True)

df_merged


# In[22]:


final_df = df_merged[['Period_x', 'AveragePrice', 'TurnoverInRs', 'TTQ/NT', 'Per_DelQty/TTQ']]
final_df


# ## Logic to display the trend as Increase, Decrease or No Change

# In[23]:


trend_df=final_df.copy()
trend_df


# In[24]:


# Add new empty columns for finding the differences

trend_df['PriceDiff'] = ''
trend_df['TurnoverDiff'] = ''
trend_df['TTQ/NTDiff'] =''
trend_df['Per_DelQty/TTQ_Diff'] =''
trend_df


# In[25]:


# Convert all columns except 'Period' to numeric
trend_df[trend_df.columns.difference(['Period_x'])] = trend_df[trend_df.columns.difference(['Period_x'])].apply(pd.to_numeric)
trend_df


# In[26]:


# Calculate difference between present row and next row
trend_df['PriceDiff'] = trend_df['AveragePrice'].diff(periods=-1)
trend_df['TurnoverDiff'] = trend_df['TurnoverInRs'].diff(periods=-1)
trend_df['TTQ/NTDiff'] = trend_df['TTQ/NT'].diff(periods=-1)
trend_df['Per_DelQty/TTQ_Diff'] = trend_df['Per_DelQty/TTQ'].diff(periods=-1)
trend_df


# In[27]:


# Create new columns based on conditions
trend_df['PriceDiff_Trend'] = trend_df['PriceDiff'].apply(lambda x: 'Increase' if x > 0 else ('No Change' if pd.isna(x) or x == 0 else 'Decrease'))
trend_df['TurnoverDiff_Trend'] = trend_df['TurnoverDiff'].apply(lambda x: 'Increase' if x > 0 else ('No Change' if pd.isna(x) or x == 0 else 'Decrease'))
trend_df['TTQ/NTDiff_Trend'] = trend_df['TTQ/NTDiff'].apply(lambda x: 'Increase' if x > 0 else ('No Change' if pd.isna(x) or x == 0 else 'Decrease'))
trend_df['Per_DelQty/TTQ_Diff_Trend'] = trend_df['Per_DelQty/TTQ_Diff'].apply(lambda x: 'Increase' if x > 0 else ('No Change' if pd.isna(x) or x == 0 else 'Decrease'))
trend_df


# ## Merge selective columns from trend_df with final_df

# In[28]:


#Append these columns to final_df
final_df['PriceDiff_Trend'] = trend_df['PriceDiff_Trend']
final_df['TurnoverDiff_Trend'] = trend_df['TurnoverDiff_Trend']
final_df['TTQ/NTDiff_Trend'] = trend_df['TTQ/NTDiff_Trend']
final_df['Per_DelQty/TTQ_Diff_Trend'] = trend_df['Per_DelQty/TTQ_Diff_Trend']
final_df


# In[29]:


df_symbol= df['Symbol']


# In[30]:


final_df['Symbol'] = df_symbol
final_df


# In[31]:


# Specify the full path where you want to save the file
file_path = r"F:\Markets\SSTS\3 5 8 13\3_5_8_13_nselib_automated_analysis.xlsx"

# Write to Excel
with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
    final_df.to_excel(writer, sheet_name='Trend')
    df.to_excel(writer, sheet_name='Raw Data', index=False)

    # Autofit column widths
    for sheet_name in writer.sheets:
        sheet = writer.sheets[sheet_name]
        for idx, col in enumerate(df.columns):
            series = df[col]
            max_len = max((
                series.astype(str).map(len).max(),
                len(str(series.name))
            )) + 1
            sheet.set_column(idx, idx, max_len)
            
print("Data has been saved to",file_path)

