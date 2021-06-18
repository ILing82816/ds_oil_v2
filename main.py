# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:45:46 2021

@author: USER
"""

import pandas as pd
from pandas import Grouper
import os
from neuralprophet import NeuralProphet
import plotly.express as px
from plotly.offline import plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math

path_prefix = 'D:/ds_oil_v2'

#@ Load Data

wti = pd.read_csv(os.path.join(path_prefix, 'Crude Oil WTI Futures Historical Data.csv'),usecols = ["Date", "Price"],parse_dates =["Date"], index_col ="Date")
wti.rename(columns={'Price':"y"}, 
                 inplace=True)
wti1 = pd.read_csv(os.path.join(path_prefix, 'Crude Oil WTI Futures Historical Data (1).csv'),usecols = ["Date", "Price"],parse_dates =["Date"], index_col ="Date")
wti1.rename(columns={'Price':"y"}, 
                 inplace=True)
data = pd.concat([wti1,wti])

#@ Data Plot

## Time Seroes Line Plot
Time_plot = px.line(data)
plot(Time_plot)

## Season Plot
groups = data.groupby(Grouper(freq='5A'))
years = pd.DataFrame()
for name, group in groups:
    years[name.year] =pd.Series(group.values.reshape(-1))
Season_plot = px.line(years)
plot(Season_plot)
    
## ACF/ PACF
plot_pacf(data.values)
plot_acf(data.values)

## Lag Plot
lags = 9
values = pd.DataFrame(data.values.reshape(-1))
columns = [values]
for i in range(1,(lags + 1)):
    columns.append(values.shift(i))
lag_data = pd.concat(columns, axis=1)
columns = ['t']
for i in range(1,(lags + 1)):
    columns.append('t-' + str(i))
lag_data.columns = columns


Lag_plot = make_subplots(rows=3, cols=3)
for i in range(1,(lags+1)):
    if i%3 == 0:
        col = 3
    else:
        col = i%3
        
    Lag_plot.add_trace(
        go.Scatter(x=lag_data["t"], y=lag_data["t-"+str(i)], mode='markers', name = "t-"+str(i)),
        row=math.ceil(i/3), col=col
    )
Lag_plot.update_layout(height=600, width=800)
plot(Lag_plot)

#@ Train

data = data.reset_index()
data = data.rename(columns={'Date':'ds'})

## Long term trend prediction
m = NeuralProphet(n_changepoints=18,
                  changepoints_range=0.9
                  )

### user specified events -- history events
election = pd.DataFrame({
  'event': 'election',
  'ds': pd.to_datetime(['1996-11-01', '2000-11-01', '2004-11-01',
                        '2008-11-01', '2012-11-01', '2016-11-01',
                        '2020-11-01'])
})
covid = pd.DataFrame({
  'event': 'covid',
  'ds': pd.to_datetime(['2020-03-09'])
})
crisis = pd.DataFrame({
  'event': 'crisis',
  'ds': pd.to_datetime(['2008-05-20'])#'2000-03-10',"2003-03-20", ,'2010-09-15'
})
holidays = pd.concat((election, covid, crisis))
m = m.add_events(["election"], lower_window=-120, upper_window=30)
m = m.add_events(["covid"], lower_window=0, upper_window=60)
m = m.add_events(['crisis'], lower_window=-120, upper_window=360)

history_df = m.create_df_with_events(data, holidays)

metrics = m.fit(history_df, freq="B", validate_each_epoch=True, valid_p=0.1,epochs=15)
future = m.make_future_dataframe(history_df, holidays, periods=365, n_historic_predictions=len(data))
forecast = m.predict(future)
fig1 = m.plot(forecast)
fig_comp = m.plot_components(forecast)

## Short term price prediction
m1 = NeuralProphet(n_changepoints=18,
                   changepoints_range=0.9,
                   n_lags=3*45,
                   n_forecasts=1*45,
                   batch_size=64,
                   epochs=10,    
                   learning_rate=1.0
                  )

### user specified events -- history events
election = pd.DataFrame({
  'event': 'election',
  'ds': pd.to_datetime(['1996-11-01', '2000-11-01', '2004-11-01',
                        '2008-11-01', '2012-11-01', '2016-11-01',
                        '2020-11-01'])
})
covid = pd.DataFrame({
  'event': 'covid',
  'ds': pd.to_datetime(['2020-03-09'])
})
crisis = pd.DataFrame({
  'event': 'crisis',
  'ds': pd.to_datetime(['2008-05-20'])#'2000-03-10',"2003-03-20", ,'2010-09-15'
})
holidays = pd.concat((election, covid, crisis))
m1 = m1.add_events(["election"], lower_window=-120, upper_window=30)
m1 = m1.add_events(["covid"], lower_window=0, upper_window=60)
m1 = m1.add_events(['crisis'], lower_window=-120, upper_window=360)

history_df = m1.create_df_with_events(data, holidays)

metrics = m1.fit(history_df, freq="B")
future = m1.make_future_dataframe(history_df, holidays, periods=365, n_historic_predictions=len(data))
forecast = m1.predict(future)
fig2 = m1.plot(forecast)
m1 = m1.highlight_nth_step_ahead_of_each_forecast(1*45) # temporary workaround to plot actual AR weights
fig3 = m1.plot_last_forecast(forecast, include_previous_forecasts=1*45)


