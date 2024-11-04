#!/usr/bin/env python
# coding: utf-8

# In[92]:


import pandas as pd
import numpy as np

# Replace this URL with the correct raw URL of your file on GitHub
url = 'https://github.com/bretts16/superbowl/blob/main/superbowl.xlsx?raw=true'

# Load the Excel file from the GitHub URL, skipping unnecessary rows and using the correct header row
try:
    last_price_data = pd.read_excel(url, sheet_name='lastprice', skiprows=3, engine='openpyxl')
    
    # Rename the first column to 'Date' and set it as the index
    last_price_data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    last_price_data.set_index('Date', inplace=True)

    # Drop any columns with 'Unnamed' in their names
    last_price_data = last_price_data.loc[:, ~last_price_data.columns.str.contains('^Unnamed')]

    # Display the first few rows to verify it loaded correctly
    print("Last Price Data:")
    print(last_price_data.head())
except Exception as e:
    print(f"Error: {e}")


# In[93]:


# Load the 'putcall' sheet from the Excel file, starting from the correct header row
put_call_data = pd.read_excel(url, sheet_name='putcall', header=3)

# Remove any rows or columns with NaN in the beginning, which may have been incorrectly interpreted as data
put_call_data.dropna(how='all', inplace=True)

# Ensure the first column is 'Dates' and set it as the index
put_call_data.rename(columns={put_call_data.columns[0]: 'Dates'}, inplace=True)
put_call_data.set_index('Dates', inplace=True)

# Display the first few rows to verify it loaded correctly
print("Put/Call Ratios Data:")
print(put_call_data.head())


# In[94]:


# Load the 'putcall' sheet from the Excel file without a header so we can manually manage it
put_call_data = pd.read_excel(url, sheet_name='putcall', header=None)

# Manually set the header and identify the start of actual data
header_row_index = 1  # Assuming row 2 is the header with company names
put_call_data.columns = put_call_data.iloc[header_row_index]

# Drop rows that are not actual data (header rows and empty ones)
put_call_data = put_call_data.iloc[header_row_index + 1:]

# Rename the first column to 'Dates' and set it as the index
put_call_data.rename(columns={put_call_data.columns[0]: 'Dates'}, inplace=True)
put_call_data.set_index('Dates', inplace=True)

# Strip any white spaces from the column names and convert them to strings
put_call_data.columns = put_call_data.columns.astype(str).str.strip()

# Convert the 'Dates' index to datetime with a specified format (assuming the format is YYYY-MM-DD)
put_call_data.index = pd.to_datetime(put_call_data.index, format='%Y-%m-%d', errors='coerce')

# Drop any rows with invalid date indices (NaT in the index)
put_call_data = put_call_data[put_call_data.index.notna()]

# Display the first few rows to verify it loaded correctly
print("Put/Call Ratios Data with Correct Column Names:")
print(put_call_data.head())


# In[95]:


# Add a trading signal column for each company in the DataFrame
tickers = put_call_data.columns.to_list()
trade_signal_df = put_call_data.copy()

for company in tickers:
    # Convert values to numeric, forcing errors to NaN
    trade_signal_df[company] = pd.to_numeric(trade_signal_df[company])
    trade_signal_df[f"{company}_Signal"] = np.where(trade_signal_df[company] > 0.7, 'Sell','Buy')

# Display the updated DataFrame to verify
trade_signal_df


# In[96]:


# Assuming 'last_price_data' is the DataFrame that contains the last price for each company on each date
last_price_df = last_price_data.copy()  # Create a copy of the last price data

# Create a new DataFrame to track the number of shares held over time for each company
shares_held_df = pd.DataFrame(index=last_price_df.index, columns=last_price_df.columns)
shares_held_df.iloc[0, :] = 1  # Start with 1 share of each company on the first date

# Create a new DataFrame to track the value of holdings for each company over time
company_value_df = pd.DataFrame(index=last_price_df.index, columns=last_price_df.columns)

# Loop through the dates to calculate the number of shares and portfolio values

            
for i in range(1, len(last_price_df)):
    current_date = last_price_df.index[i]
    previous_date = last_price_df.index[i - 1]

    # Carry forward the number of shares held from the previous day
    shares_held_df.loc[current_date] = shares_held_df.loc[previous_date]

    # Check if the current date exists in trade_signal_df
    if current_date in trade_signal_df.index:
        for company in last_price_df.columns:
            signal = trade_signal_df.loc[current_date, f"{company}_Signal"]

            if signal == 'Buy':
                shares_held_df.loc[current_date, company] += 1  # Buy one additional share
            elif signal == 'Sell' and shares_held_df.loc[current_date, company] > 0:
                shares_held_df.loc[current_date, company] -= 1  # Sell one share if possible
    else:
        print(f"Date {current_date} not found in trade_signal_df")

    # Calculate the value of holdings for each company for the current date
    company_value_df.loc[current_date] = shares_held_df.loc[current_date] * last_price_df.loc[current_date]           
            
            
            

    # Calculate the value of holdings for each company for the current date
    company_value_df.loc[current_date] = shares_held_df.loc[current_date] * last_price_df.loc[current_date]

# Create a new column for the total portfolio value for each date
company_value_df['Total_Portfolio_Value'] = company_value_df.sum(axis=1)

# Display the updated portfolio value DataFrame to verify the results
company_value_df


# In[97]:


# Create a new DataFrame to store the updated number of shares held for each company over time
shares_held_dynamic_df = pd.DataFrame(index=trade_signal_df.index, columns=tickers)

# Initialize with 1 share for each company on the first date
shares_held_dynamic_df.iloc[0] = 1

# Iterate over each date to determine the number of shares held based on trading signals
for i in range(1, len(trade_signal_df)):
    for company in tickers:
        # Carry forward the number of shares from the previous day
        shares_held_dynamic_df.loc[trade_signal_df.index[i], company] = shares_held_dynamic_df.loc[trade_signal_df.index[i - 1], company]
        
        # If the trading signal is "Buy", add one share
        if trade_signal_df[f"{company}_Signal"].iloc[i] == 'Buy':
            shares_held_dynamic_df.loc[trade_signal_df.index[i], company] += 1
        # If the trading signal is "Sell", subtract one share (but not going below 0)
        elif trade_signal_df[f"{company}_Signal"].iloc[i] == 'Sell':
            shares_held_dynamic_df.loc[trade_signal_df.index[i], company] = max(0, shares_held_dynamic_df.loc[trade_signal_df.index[i], company] - 1)

# Create a new DataFrame to store the portfolio values for each company
portfolio_value_dynamic_df = shares_held_dynamic_df.copy()

# Calculate the value of holdings by multiplying shares held by the last price for each date
for company in last_price_df.columns:
    portfolio_value_dynamic_df[company] = shares_held_dynamic_df[company] * last_price_df[company]

# Create a new column to calculate the total portfolio value on each date
portfolio_value_dynamic_df['Total_Portfolio_Value'] = portfolio_value_dynamic_df.sum(axis=1)

# Display the updated portfolio value DataFrame
portfolio_value_dynamic_df


# In[98]:


# Assume 'trade_signal_df' and 'last_price_data' are already loaded correctly.

# Initialize a new DataFrame to track the number of shares held for each company
shares_held_df = pd.DataFrame(0, index=trade_signal_df.index, columns=last_price_data.columns)

# Initialize a new DataFrame to track portfolio value over time
portfolio_value_df = pd.DataFrame(0.0, index=trade_signal_df.index, columns=last_price_data.columns)

# Iterate over each date to calculate shares held and portfolio value
for date in trade_signal_df.index:
    if date == trade_signal_df.index[0]:
        # On the first date, buy or short as per trading signal
        for company in last_price_data.columns:
            signal = trade_signal_df.loc[date, f"{company}_Signal"]
            price = last_price_data.loc[date, company]
            
            if signal == 'Buy':
                shares_held_df.loc[date, company] += 1  # Buy 1 share
            elif signal == 'Short':
                shares_held_df.loc[date, company] -= 1  # Short 1 share
            
            # Calculate portfolio value based on current shares held
            portfolio_value_df.loc[date, company] = shares_held_df.loc[date, company] * price
    else:
        # On subsequent dates, carry forward previous holdings and apply new trading signals
        prev_date = trade_signal_df.index[trade_signal_df.index.get_loc(date) - 1]
        shares_held_df.loc[date] = shares_held_df.loc[prev_date]  # Start with previous day's holdings
        
        for company in last_price_data.columns:
            signal = trade_signal_df.loc[date, f"{company}_Signal"]
            price = last_price_data.loc[date, company]
            
            if signal == 'Buy':
                shares_held_df.loc[date, company] += 1  # Buy 1 share
            elif signal == 'Short':
                shares_held_df.loc[date, company] -= 1  # Short 1 share
            
            # Calculate portfolio value based on current shares held
            portfolio_value_df.loc[date, company] = shares_held_df.loc[date, company] * price

# Calculate total portfolio value by summing across all companies
portfolio_value_df['Total_Portfolio_Value'] = portfolio_value_df.sum(axis=1)
portfolio_value_df = portfolio_value_df.dropna() 

# Display the resulting DataFrames
print("Shares Held Over Time:")
print(shares_held_df.head())

print("\nPortfolio Value Over Time:")
print(portfolio_value_df.tail())


# In[99]:


# Calculate the total dollar value at the end of the timeframe
total_dollar_value = portfolio_value_df['Total_Portfolio_Value'].iloc[-1]

print(f"The total dollar value of the portfolio at the end of the timeframe is: ${total_dollar_value:.2f}")
portfolio_value_df


# In[100]:


import matplotlib.pyplot as plt
# Calculate the daily returns as the percentage change in Total_Portfolio_Value
portfolio_value_df['Daily_Return'] = portfolio_value_df['Total_Portfolio_Value'].pct_change()

# Calculate cumulative returns based on daily returns, assuming initial investment of 1 (or you could use any starting value)
portfolio_value_df['Cumulative_Return'] = (1 + portfolio_value_df['Daily_Return']).cumprod()

# Plot the Cumulative Return over time
plt.figure(figsize=(10, 6))
plt.plot(portfolio_value_df.index, portfolio_value_df['Cumulative_Return'], label='Cumulative Return', linewidth=2, color='green')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Cumulative Portfolio Return Over Time')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


# In[101]:


first_day_value = (last_price_data.iloc[0] * 1).sum()
print(f'Portfolio value on the first day: ${first_day_value:.2f}')


# In[102]:


# Get the portfolio value on the second last date in the dataset
second_last_date = last_price_data.index[-2]
second_last_day_value = (last_price_data.loc[second_last_date] * 1).sum()
print(f'Portfolio value on the second last date ({second_last_date}): ${second_last_day_value:.2f}')


# In[103]:


# Print the value of the portfolio at each point in time
print("Portfolio Value at Each Point in Time:")
print(portfolio_value_df['Total_Portfolio_Value'])

# Print the value for the first week
print("\nPortfolio Value for the First Week:")
print(portfolio_value_df['Total_Portfolio_Value'].head(7))

# Print the value for the last week
print("\nPortfolio Value for the Last Week:")
print(portfolio_value_df['Total_Portfolio_Value'].tail(7))


# In[104]:


# Load the 's&p' sheet from the Excel file
try:
    sp_data = pd.read_excel(url, sheet_name='s&p', engine='openpyxl')
    
    # Rename the first column to 'Date' and set it as the index
    sp_data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    sp_data.set_index('Date', inplace=True)

    # Drop the column that says '% Returns'
    if '% Returns' in sp_data.columns:
        sp_data.drop(columns=['% Returns'], inplace=True)

    # Display the first few rows to verify it loaded correctly
    print("S&P Data:")
    print(sp_data.head())
except Exception as e:
    print(f"Error: {e}")

# Calculate the cumulative returns for the S&P data
initial_investment = 8678.47

# Calculate daily returns
sp_data['Daily_Return'] = sp_data.iloc[:, 0].pct_change()

# Calculate cumulative returns
sp_data['Cumulative_Return'] = (1 + sp_data['Daily_Return']).cumprod() * initial_investment

# Display the cumulative returns
print("Cumulative Returns for the S&P:")
print(sp_data[['Cumulative_Return']])


# In[105]:


# Load the Excel file from the GitHub URL
last_price_data = pd.read_excel(url, sheet_name='lastprice', skiprows=3, engine='openpyxl')
last_price_data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
last_price_data.set_index('Date', inplace=True)
last_price_data = last_price_data.loc[:, ~last_price_data.columns.str.contains('^Unnamed')]
last_price_data.index = pd.to_datetime(last_price_data.index, format='%Y-%m-%d', errors='coerce')
last_price_data = last_price_data.dropna()

# Initialize dataframes for tracking shares and portfolio values
shares_held_df = pd.DataFrame(0, index=trade_signal_df.index, columns=last_price_data.columns)
portfolio_value_df = pd.DataFrame(0.0, index=trade_signal_df.index, columns=last_price_data.columns)

# Set the initial investment amount
initial_investment = 8678.0
cash_balance = initial_investment  # Start with the full initial investment as cash

# Loop through each date and adjust holdings based on signals and available cash
for date in trade_signal_df.index:
    daily_value = 0  # Track the total value of the portfolio for the day
    if date == trade_signal_df.index[0]:  # First day
        for company in last_price_data.columns:
            signal = trade_signal_df.loc[date, f"{company}_Signal"]
            price = last_price_data.loc[date, company] if date in last_price_data.index and company in last_price_data.columns else 0
            if signal == 'Buy' and cash_balance >= price:
                shares_held_df.loc[date, company] = 1  # Buy one share
                cash_balance -= price
            elif signal == 'Short' and cash_balance >= price:
                shares_held_df.loc[date, company] = -1  # Short one share
                cash_balance -= price
            portfolio_value_df.loc[date, company] = shares_held_df.loc[date, company] * price
            daily_value += portfolio_value_df.loc[date, company]
    else:  # For all other days
        prev_date = trade_signal_df.index[trade_signal_df.index.get_loc(date) - 1]
        shares_held_df.loc[date] = shares_held_df.loc[prev_date]  # Carry over previous holdings
        for company in last_price_data.columns:
            signal = trade_signal_df.loc[date, f"{company}_Signal"]
            price = last_price_data.loc[date, company] if date in last_price_data.index and company in last_price_data.columns else 0
            if signal == 'Buy' and cash_balance >= price:
                shares_held_df.loc[date, company] += 1
                cash_balance -= price
            elif signal == 'Short' and cash_balance >= price:
                shares_held_df.loc[date, company] -= 1
                cash_balance -= price
            portfolio_value_df.loc[date, company] = shares_held_df.loc[date, company] * price
            daily_value += portfolio_value_df.loc[date, company]
    # Update the total portfolio value for the day
    portfolio_value_df.loc[date, 'Total_Portfolio_Value'] = daily_value + cash_balance

# Drop the last date to avoid plotting a drop to zero
portfolio_value_df = portfolio_value_df.iloc[:-1]

# Normalize the portfolio cumulative value to start at the initial investment
portfolio_value_df['Normalized_Portfolio_Value'] = portfolio_value_df['Total_Portfolio_Value'] / initial_investment * initial_investment

# Load the S&P data and calculate its normalized cumulative value starting at initial investment
sp_data = pd.read_excel(url, sheet_name='s&p', engine='openpyxl')
sp_data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
sp_data.set_index('Date', inplace=True)
sp_data.index = pd.to_datetime(sp_data.index, format='%Y-%m-%d', errors='coerce')
sp_data = sp_data.dropna()

# Normalize the S&P cumulative value to start at the initial investment
sp_data['Normalized_S&P_Value'] = sp_data.iloc[:, 0] / sp_data.iloc[0, 0] * initial_investment

# Drop the last date from S&P data if it has the same issue (optional, for consistency)
sp_data = sp_data.iloc[:-1]

# Plot the cumulative values over time for both the portfolio and the S&P
plt.figure(figsize=(10, 6))
plt.plot(portfolio_value_df.index, portfolio_value_df['Normalized_Portfolio_Value'], label='Portfolio Cumulative Value', linewidth=2, color='green')
plt.plot(sp_data.index, sp_data['Normalized_S&P_Value'], label='S&P Cumulative Value', linewidth=2, color='blue')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Cumulative Value ($)')
plt.title('Cumulative Value Over Time: Portfolio vs. S&P (Starting from $8,678)')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()


# In[106]:


# Align portfolio and S&P returns and drop any rows with missing data
combined_df = pd.DataFrame({'Portfolio_Returns': portfolio_returns, 'SP_Returns': sp_returns}).dropna()

# Check for extreme values or zero returns that may skew the regression
combined_df = combined_df[(combined_df['Portfolio_Returns'] != 0) & (combined_df['SP_Returns'] != 0)]

# Set up the regression model with aligned data
X = combined_df['SP_Returns']
Y = combined_df['Portfolio_Returns']
X = sm.add_constant(X)

# Run the regression
model = sm.OLS(Y, X).fit()
print(model.summary())

# Optional: Calculate rolling beta to observe changes over time
rolling_window = 60  # 60-day rolling window
combined_df['Rolling_Beta'] = (
    combined_df['Portfolio_Returns'].rolling(window=rolling_window).corr(combined_df['SP_Returns'])
)

# Plot the rolling beta
plt.figure(figsize=(10, 6))
plt.plot(combined_df.index, combined_df['Rolling_Beta'], label='Rolling Beta (60-day)', color='purple')
plt.xlabel('Date')
plt.ylabel('Beta')
plt.title('Rolling Beta of Portfolio vs. S&P 500')
plt.legend()
plt.grid(True)
plt.show()


# In[107]:


# Define Super Bowl dates and trading weeks
super_bowl_dates = {
    2020: '2020-02-03',
    2021: '2021-02-08',
    2022: '2022-02-14',
    2023: '2023-02-13',
    2024: '2024-02-12'
}

# Convert dates to actual trading weeks (Monday through Friday after each Super Bowl)
trading_weeks = {}
for year, date in super_bowl_dates.items():
    start_date = pd.to_datetime(date)
    trading_weeks[year] = pd.date_range(start=start_date, periods=5, freq='B')  # 5 business days

# Initialize DataFrames for tracking shares and portfolio values
shares_held_df = pd.DataFrame(0, index=trade_signal_df.index, columns=last_price_data.columns)
portfolio_value_df = pd.DataFrame(0.0, index=trade_signal_df.index, columns=last_price_data.columns)

# Set the initial investment amount
initial_investment = 8678.0
cash_balance = initial_investment  # Start with full investment as cash
total_value_history = []  # To keep track of total value over time

# Loop through each date and execute trades during Super Bowl week, holding cash otherwise
for date in trade_signal_df.index:
    # Check if this date is within a Super Bowl trading week
    trading_active = any(date in week for week in trading_weeks.values())
    
    if trading_active:
        daily_value = 0  # Track total portfolio value for the day
        for company in last_price_data.columns:
            signal = trade_signal_df.loc[date, f"{company}_Signal"]
            price = last_price_data.loc[date, company] if date in last_price_data.index and company in last_price_data.columns else 0
            
            if signal == 'Buy' and cash_balance >= price:
                shares_held_df.loc[date, company] += 1
                cash_balance -= price
            elif signal == 'Short' and cash_balance >= price:
                shares_held_df.loc[date, company] -= 1
                cash_balance -= price
            
            portfolio_value_df.loc[date, company] = shares_held_df.loc[date, company] * price
            daily_value += portfolio_value_df.loc[date, company]
        
        # Update the total value to include cash balance
        total_value = daily_value + cash_balance
    else:
        # If it's outside of Super Bowl week, just carry the last portfolio value
        total_value = total_value_history[-1] if total_value_history else initial_investment
    
    # Append to the history and set for plotting
    total_value_history.append(total_value)
    portfolio_value_df.loc[date, 'Total_Portfolio_Value'] = total_value

# Create a normalized cumulative value column for plotting
portfolio_value_df['Normalized_Portfolio_Value'] = portfolio_value_df['Total_Portfolio_Value'] / initial_investment * initial_investment

# Load and process S&P data for comparison
sp_data = pd.read_excel(url, sheet_name='s&p', engine='openpyxl')
sp_data.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
sp_data.set_index('Date', inplace=True)
sp_data.index = pd.to_datetime(sp_data.index, format='%Y-%m-%d', errors='coerce')
sp_data = sp_data.dropna()

# Normalize the S&P cumulative value to start at the initial investment
sp_data['Normalized_S&P_Value'] = sp_data.iloc[:, 0] / sp_data.iloc[0, 0] * initial_investment

# Plot the cumulative values over time for both the portfolio and the S&P
plt.figure(figsize=(10, 6))
plt.plot(portfolio_value_df.index, portfolio_value_df['Normalized_Portfolio_Value'], label='Portfolio Cumulative Value (Super Bowl Week Only)', linewidth=2, color='green')
plt.plot(sp_data.index, sp_data['Normalized_S&P_Value'], label='S&P Cumulative Value', linewidth=2, color='blue')

# Add labels and title
plt.xlabel('Date')
plt.ylabel('Cumulative Value ($)')
plt.title('Cumulative Value Over Time: Portfolio (Super Bowl Week Only) vs. S&P (Starting from $8,678)')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()



# In[ ]:




