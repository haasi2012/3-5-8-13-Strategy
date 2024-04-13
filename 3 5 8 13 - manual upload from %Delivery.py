#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import datetime as dt
import zipfile
from io import BytesIO, StringIO
from nselib.libutil import *
from nselib.constants import *
from nselib import capital_market
import nselib


# In[2]:


# Read csv file

df = pd.read_csv(r"F:\Markets\SSTS\3 5 8 13\Stock.csv", skipinitialspace = True)
df.columns = df.columns.str.strip() #remove blank from column headers
df.tail()


# In[3]:


#Consider only Equities related data
df = df[df['Series'] == 'EQ']
df


# In[4]:


# Remove commas and convert to integer
df['No. of Trades'] = df['No. of Trades'].str.replace(',', '').astype(int)
df['Total Traded Quantity'] = df['Total Traded Quantity'].str.replace(',', '').astype(int)
df['Deliverable Qty'] = df['Deliverable Qty'].str.replace(',', '').astype('float64')
df['Turnover ₹'] = df['Turnover ₹'].str.replace(',', '').astype('float64')
# Remove commas from the 'Total Traded Quantity' column and convert to float64 format
df['Total Traded Quantity'] = df['Total Traded Quantity'].astype('float64')
df['No. of Trades'] = df['No. of Trades'].astype('float64')
print(df.dtypes)


# In[5]:


# Calculate TTQ/NT
df['TTQ/NT'] = df['Total Traded Quantity'] / df['No. of Trades']
df


# In[6]:


# Select only relevant columns for analysis
df1 = df[['Symbol', 'Date','Average Price', 'Total Traded Quantity', 'Turnover ₹', 'No. of Trades', 'Deliverable Qty', '% Dly Qt to Traded Qty', 'TTQ/NT']]
df1


# In[7]:


# Calculate the sums for the last 3 days, 5 days, 8 days and 13 days for calculating percentile of delivered quantity to traded quantity
df2 = pd.DataFrame({
    'Last 3 Days': [
        df1['Deliverable Qty'].iloc[-3:].sum(),
        df1['Total Traded Quantity'].iloc[-3:].sum()
    ],
    'Last 5 Days': [
        df1['Deliverable Qty'].iloc[-5:].sum(),
        df1['Total Traded Quantity'].iloc[-5:].sum()
    ],
    'Last 8 Days': [
        df1['Deliverable Qty'].iloc[-8:].sum(),
        df1['Total Traded Quantity'].iloc[-8:].sum()
    ],
    'Last 13 Days': [
        df1['Deliverable Qty'].iloc[-13:].sum(),
        df1['Total Traded Quantity'].iloc[-13:].sum()
    ]
}, index=['Sum of Deliverable Qty', 'Sum of Total Traded Quantity'])

# Display the new dataframe 'df2'
df2


# In[8]:


# Create a DataFrame
df_t = pd.DataFrame(df2)

# Transpose the DataFrame and add a heading to the first column
df2 = df_t.T.reset_index()
df2.columns = ['Period', 'Sum of Deliverable Qty', 'Sum of Total Traded Quantity']
df2


# In[9]:


# Calculate the percentage of deliverable quantity to total traded quantity
df2['Per_DelQty/TTQ'] = df2['Sum of Deliverable Qty'] / df2['Sum of Total Traded Quantity'] * 100
df2


# In[10]:


df2 = df2.drop(['Sum of Deliverable Qty', 'Sum of Total Traded Quantity'], axis=1)
df2


# In[11]:


# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])


# In[12]:


# Sort dataframe by date
df = df.sort_values(by='Date')


# In[13]:


# Select required columns
selected_columns = ['Average Price', 'Total Traded Quantity', 'Turnover ₹', 'No. of Trades', 'Deliverable Qty', 'TTQ/NT']
selected_df = df[selected_columns]
print(selected_df.dtypes)


# In[14]:


#Convert object to float64
selected_df['No. of Trades'] = pd.to_numeric(selected_df['No. of Trades'], errors='coerce')
selected_df


# In[15]:


# Calculate rolling means for the specified days
rolling_means = pd.DataFrame()
rolling_means['Last 3 Days'] = selected_df.rolling(window=3).mean().iloc[-1]
rolling_means['Last 5 Days'] = selected_df.rolling(window=5).mean().iloc[-1]
rolling_means['Last 8 Days'] = selected_df.rolling(window=8).mean().iloc[-1]
rolling_means['Last 13 Days'] = selected_df.rolling(window=13).mean().iloc[-1]

rolling_means


# In[16]:


# Transpose the DataFrame
transposed_means = rolling_means.transpose()

# Assign the title 'Period' to the first column
transposed_means.columns.name = 'Period'

transposed_means


# In[17]:


rolling_means = transposed_means
rolling_means


# In[18]:


# Convert the DataFrame to integer type, specifying int64 for Turnover
rolling_means_filled_whole = rolling_means.astype({
    'Average Price': int, 
    'Total Traded Quantity': int, 
    'Turnover ₹': np.int64, 
    'No. of Trades': int, 
    'Deliverable Qty': int, 
})

df1 = rolling_means_filled_whole
df1


# In[19]:


print(df1.columns)


# In[20]:


print(df2.columns)


# In[21]:


# Reset the index of df1 and rename the index column to 'Period'
df1_reset = df1.reset_index().rename(columns={'index': 'Period'})

# Merge df1_reset and df2 based on the 'Period' column
df_merged = pd.merge(df1_reset, df2, on='Period')
df_merged


# In[22]:


final_df = df_merged[['Period', 'Average Price', 'Turnover ₹', 'TTQ/NT', 'Per_DelQty/TTQ']]
final_df


# In[23]:


df['Symbol']


# In[24]:


final_df


# ## Logic to display the trend as Increase, Decrease or No Change

# In[25]:


trend_df=final_df.copy()


# In[26]:


trend_df


# In[27]:


# Add new empty columns for finding the differences

trend_df['PriceDiff'] = ''
trend_df['TurnoverDiff'] = ''
trend_df['TTQ/NTDiff'] =''
trend_df['Per_DelQty/TTQ_Diff'] =''
trend_df


# In[28]:


# Convert all columns except 'Period' to numeric
trend_df[trend_df.columns.difference(['Period'])] = trend_df[trend_df.columns.difference(['Period'])].apply(pd.to_numeric)
trend_df


# In[29]:


# Calculate difference between present row and next row
trend_df['PriceDiff'] = trend_df['Average Price'].diff(periods=-1)
trend_df['TurnoverDiff'] = trend_df['Turnover ₹'].diff(periods=-1)
trend_df['TTQ/NTDiff'] = trend_df['TTQ/NT'].diff(periods=-1)
trend_df['Per_DelQty/TTQ_Diff'] = trend_df['Per_DelQty/TTQ'].diff(periods=-1)
trend_df


# In[30]:


# Create new columns based on conditions
trend_df['PriceDiff_Trend'] = trend_df['PriceDiff'].apply(lambda x: 'Increase' if x > 0 else ('No Change' if pd.isna(x) or x == 0 else 'Decrease'))
trend_df['TurnoverDiff_Trend'] = trend_df['TurnoverDiff'].apply(lambda x: 'Increase' if x > 0 else ('No Change' if pd.isna(x) or x == 0 else 'Decrease'))
trend_df['TTQ/NTDiff_Trend'] = trend_df['TTQ/NTDiff'].apply(lambda x: 'Increase' if x > 0 else ('No Change' if pd.isna(x) or x == 0 else 'Decrease'))
trend_df['Per_DelQty/TTQ_Diff_Trend'] = trend_df['Per_DelQty/TTQ_Diff'].apply(lambda x: 'Increase' if x > 0 else ('No Change' if pd.isna(x) or x == 0 else 'Decrease'))
trend_df


# ## Merge selective columns from trend_df with final_df

# In[31]:


#Append these columns to final_df
final_df['PriceDiff_Trend'] = trend_df['PriceDiff_Trend']
final_df['TurnoverDiff_Trend'] = trend_df['TurnoverDiff_Trend']
final_df['TTQ/NTDiff_Trend'] = trend_df['TTQ/NTDiff_Trend']
final_df['Per_DelQty/TTQ_Diff_Trend'] = trend_df['Per_DelQty/TTQ_Diff_Trend']
final_df


# In[32]:


df_symbol= df['Symbol']


# In[33]:


final_df['Symbol'] = df_symbol
final_df


# In[34]:


# Specify the full path where you want to save the file
file_path = r"F:\Markets\SSTS\3 5 8 13\3_5_8_13_manual_analysis.xlsx"

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

