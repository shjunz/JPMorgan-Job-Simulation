#!/usr/bin/env python
# coding: utf-8

# # Pricing Model

# In[ ]:


import pandas as pd


# In[30]:


def pricing_model(
    inj_dates,
    wd_dates,
    inj_prices,
    wd_prices,
    rate,
    max_volume,
    storage_fee
    ):
    """
    Calculate the value of a storage contract.

    Parameters:
    inj_dates (list): List of injection dates
    wd_dates (list): List of withdrawal dates
    inj_prices (list): List of injection prices
    wd_prices (list): List of withdrawal prices
    rate (float): Injection/Withdrawal rate
    max_volume (float): Maximum storage volume
    storage_fee (float): Storage fee

    Return:
    price (float): Value of the storage contract
    """

    # Create a list of events
    events = []
    for d in inj_dates:
        events.append((pd.to_datetime(d), +rate))
    for d in wd_dates:
        events.append((pd.to_datetime(d), -rate))

    def by_date(event):
        return event[0]

    events.sort(key = by_date)

    print(f"{'Date':<12}  {'Change':>8}")
    print('-' * 22)
    for date, change in events:
        print(f"{date.strftime('%Y-%m-%d'):<12}  {change:>+8.0f}")

    # Check storage volume at each event
    storage = 0
    for (date, change) in events:
        storage += change
        if storage > max_volume:
            raise ValueError(f"Storage capacity exceeded on {date}")
        if storage < 0:
            raise ValueError(f"Negative storage on {date}")

    if storage != 0:
        raise ValueError(f"Contract ends with {storage} MMBtu unsold in storage")

    # Calculate the value of the contract
    inj_value = rate * sum(inj_prices)
    wd_value = rate * sum(wd_prices)

    # Storage fee
    duration_months = (events[-1][0] - events[0][0]).days / 30

    storage_cost = storage_fee * duration_months

    # Final price
    price = wd_value - inj_value - storage_cost

    return price


# # Testing

# In[31]:


# Import predicted prices from Task 1
prices_df = pd.read_csv('Nat_Gas_Predicted.csv', parse_dates=['Dates'])
prices_df = prices_df.set_index('Dates')

# Function to get price for a specific date
def get_price(date):
    return prices_df.loc[pd.to_datetime(date), 'Prices']


# In[32]:


# Test 1: Basic Test

inj_dates = ['2024-06-01', '2024-07-01', '2024-08-01']
wd_dates  = ['2024-12-01', '2025-01-01', '2025-02-01']

inj_prices = [get_price(d) for d in inj_dates]
wd_prices  = [get_price(d) for d in wd_dates]

print("Injection Dates & Prices：")
for d, p in zip(inj_dates, inj_prices):
    print(f"  {d}: ${p:.2f}")
print("\nWithdrawal Dates & Prices：")
for d, p in zip(wd_dates, wd_prices):
    print(f"  {d}: ${p:.2f}")
print()

value = pricing_model(
    inj_dates, wd_dates,
    inj_prices, wd_prices,
    rate=100_000,
    max_volume=500_000,
    storage_fee=10_000,
)
print(f"\nContract Price: ${value:,.2f}")


# In[34]:


# Test 2: Alternative Injection and Withdrawal

# First round
inj_dates = ['2024-05-01', '2024-05-02']    
wd_dates  = ['2024-07-01']

# Second round
inj_dates += ['2024-09-01', '2024-09-02']
wd_dates  += ['2025-01-01', '2025-01-02', '2025-01-03']   

inj_prices = [get_price(d) for d in inj_dates]
wd_prices  = [get_price(d) for d in wd_dates]

print("Injection Dates & Prices:")
for d, p in zip(inj_dates, inj_prices):
    print(f"  {d}: ${p:.2f}")
print("\nWithdrawal Dates & Prices:")
for d, p in zip(wd_dates, wd_prices):
    print(f"  {d}: ${p:.2f}")
print()

value = pricing_model(
    inj_dates, wd_dates,
    inj_prices, wd_prices,
    rate=100_000,
    max_volume=500_000,
    storage_fee=10_000,
)
print(f"\nContract Price: ${value:,.2f}")


# In[35]:


# Test 3: Warning Messages

# 3a) max_volume
try:
    pricing_model(
        inj_dates  = ['2024-06-01', '2024-07-01', '2024-08-01'],
        wd_dates   = ['2024-12-01', '2025-01-01', '2025-02-01'],
        inj_prices = [10, 10, 10],
        wd_prices  = [13, 13, 13],
        rate       = 100_000,
        max_volume = 150_000,
        storage_fee= 10_000,
    )
except ValueError as e:
    print()
    print(f"{e}")


# In[38]:


# 3b) Negative storage
try:
    pricing_model(
        inj_dates  = ['2024-12-01'],
        wd_dates   = ['2024-06-01'],
        inj_prices = [10],
        wd_prices  = [13],
        rate       = 100_000,
        max_volume = 500_000,
        storage_fee= 10_000,
    )
except ValueError as e:
    print()
    print(f"{e}")


# In[39]:


# 3c) Unbalanced inject/withdraw
try:
    pricing_model(
        inj_dates  = ['2024-06-01', '2024-07-01', '2024-08-01'],
        wd_dates   = ['2024-12-01', '2025-01-01'],
        inj_prices = [10, 10, 10],
        wd_prices  = [13, 13],
        rate       = 100_000,
        max_volume = 500_000,
        storage_fee= 10_000,
    )
except ValueError as e:
    print()
    print(f"{e}")

