# Bike-Sharing-Demand-Prediction

<a href="#"><img width="100%" height="300" src="https://img.freepik.com/premium-vector/city-bike-rental-bicycle-rental-electronic-system-people-rent-bicycling-smart-service-cartoon-illustration_169479-332.jpg" height="175px"/></a>

##### -- Project Status: [Completed]

## Problem Statement

Currently Rental bikes are introduced in many urban cities for the enhancement of mobility comfort. It is important to make the rental bike available and accessible to the public at the right time as it lessens the waiting time. Eventually, providing the city with a stable supply of rental bikes becomes a major concern. The crucial part is the prediction of bike count required at each hour for the stable supply of rental bikes.

## Dataset Discription

### Features

* Date : year-month-day
* Hour - Hour of the day (0 to 23)
* Temperature-Temperature in Celsius
* Humidity - in %
* Windspeed - in m/s
* Visibility - for 10m
* Dew point temperature - in Celsius
* Solar radiation - in MJ/m2
* Rainfall - in mm
* Snowfall - in cm
* Seasons - Winter, Spring, Summer, Autumn
* Holiday - Holiday, No holiday
* Functional Day - Yes, No

### Output Variable

* Rented Bike count - Count of bikes rented at each hour  

## Project Workflow
1) Importing dataset and basic python libraries
2) Visualising the Data
3) Data Preparation
4) Replacing outliers with KNN imputer.
5) Splitting the Data into Training and Testing Sets
6) Feature Scaling on the train data
7) Building the Models
8) Hyparameter Tunning
9) Comparing Model Evaluation Metrics
10) Final model selection
11) Conclusion 


## Conclusion

In our analysis, first we did EDA on the dataset. We found the impact of each feature on Bike Rent Count. We found the seasonwise Bike Rent Count, Hourly Trend, Monthly Trend and Temperature wise rent count trend. As we were expecting the Bike Rent Count should be at highest at no Rainfall , no Snowfall and high Visibility, we got the same results in the EDA.

For deploying most of the ML models we want our features to be normally distributed. First, We found out that our target variable 'Bike Rent Count' has 158 outliers with the help of IQR. Such outliers make high impact on our model learning. Hence we replaced them with NaN and imputed those same ones with KNN. Squared root transform gave us the normal distribution of our target variable.

For the features Rainfall, Snowfall, Solar Radiation and Visibility getting the normal distribution was nearly impossible, beacause they either had greater positive or negative skew value. Fot those features we converted them to Categorical features. We replaced those observations either into 0 or 1.

Second IMP assumption of ML model is that we dont want any multicollinear features. we found out Dew Point Temperature and Temperature were highly co-related. As Temperature feature was making high impact on our target variable, we dropped Dew Point Temperature feature. Varience Inflation Factor for our features were below 5. With MinMaxScaler we normalized our features.

Next we implemented 7 different ML models such as Linear Regression, Lasso Regression, Ridge Regression, Elastic Net Regression, Decision Tree Regressor, Random Forest Regressor and Gradient Boost Regressor. We also used Hyper Tunning and Cross Validation for getting better results in Decision Tree Regressor. We got the best results for Random Forest Regressor. For Training set r2_score is .98 and Testing set .87 r2_score, that means 87% of the variation in the output variable is explained by the input variables.
