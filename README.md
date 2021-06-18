# Data Science Crude Oil WTI Price Estimator -- V2: Project Overview
* Created a tool that estimates data science crude oil wti price (MAE ~$5) to help investors realize oil price in the future when they invest oil-related stocks.
* Engineered features from the economic indicators to de-trend the value put on python, excel.
* Optimized Linear, Long Short-term Memory, and Prophet tuning parameters, seasonality and add_regressor, to reach the best model.

## Code and Resources Used
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, statsmodels, torch, NeuralProphet, matplotlib, seaborn  
**For Web Framework Requirements:** `pip install -r requirements.txt`  
**Model Building Github:** https://github.com/ourownstory/neural_prophet  


## Data Collecting
With each date, we got the following:
* WTI Crude Oil Price
https://www.investing.com/commodities/crude-oil-historical-data

## Data Cleaning
After collecting the data, I needed to clean it up so that it was usable for our model. I made the following changes and created the following variables:
* Parsed datetime data out of date
* Filled the previous data in missing data.

## EDA
I looked at the distributions and autocorrelation of the data, the correlation with the various variables. Below are a few highlights from the figures.  
WTI Price:  
![alt text](https://github.com/ILing82816/ds_oil_v2/blob/master/time_plot.png "Time_Series_data")  
Autocorrelation of WTI Price: There are AR(3)
![alt text](https://github.com/ILing82816/ds_oil_price_proj/blob/master/Figure/ACF_PACF.png "ACF")  


## Model Building and Performance
Depend on the trend of oil price in the future, investors decide the strategies of investment. Although the Linear Regression model far outperformed the other approaches on the test and validation sets, the Prophet model is more practical.
* **NeuralProphet:** MAE = 5.17   
![alt text](https://github.com/ILing82816/ds_oil_v2/blob/master/Predict.png "NeuralProphet")  

* **NeuralProphet -- Trend:**  
![alt text](https://github.com/ILing82816/ds_oil_v2/blob/master/Trend.png "trend")

* **NeuralProphet -- Prediction of oil price:**  
![alt text](https://github.com/ILing82816/ds_oil_v2/blob/master/Short_predict.png "Short_term")
