# OLORUNFEMI-MODUPE-PEACE_3MTT_CAPSTONE-PROJECT_COHORT2_2024

# PROJECT TOPIC: Predictive Modeling for COVID-19 in Public Health

## Introduction
The COVID-19 pandemic, caused by the SARS-CoV-2 virus, emerged in late 2019 and quickly spread worldwide, leading to a global health crisis. Characterized by respiratory symptoms ranging from mild to severe, the virus significantly impacted healthcare systems, economies, and daily life. With millions of confirmed cases and fatalities, the pandemic underscored the critical need for effective public health strategies, timely data analysis, and predictive modeling to mitigate its effects and prevent future outbreaks.

## Project Overview
The COVID-19 pandemic has presented significant challenges for public health organizations in predicting virus spread, understanding transmission factors, and optimizing resource allocation. This project, conducted for HealthGuard Analytics, leverages historical COVID-19 data to build a predictive modeling system. The analysis involves data cleaning, exploratory data analysis (EDA), and predictive modeling to forecast trends, identify risk factors, and support data-driven public health policies. The insights aim to inform strategies for outbreak management, resource distribution, and improved health outcomes.

## Data Design
The COVID-19 dataset CSV file used for this analysis was downloaded from Kaggle. It contains data from January 2020 to July 2020, with 49,068 observations and 10 columns. Below is a description of each column in the dataset:
1.	**Province/State**: Sub-national administrative region (e.g., states, provinces).
2.	**Country/Region**: Country or territory where the data was recorded.
3.	**Lat**: Latitude of the location (geographical coordinate).
4.	**Long**: Longitude of the location (geographical coordinate).
5.	**Date**: Date when the data was recorded.
6.	**Confirmed**: Cumulative number of confirmed COVID-19 cases.
7.	**Deaths**: Cumulative number of deaths due to COVID-19.
8.	**Recovered**: Cumulative number of recoveries from COVID-19.
9.	**Active**: Current active cases (calculated as Confirmed - Deaths - Recovered).
10.	**WHO Region**: World Health Organization region associated with the country/region.
The dataset was analyzed using Python.

## Data Preprocessing
To improve data quality, enhance model accuracy, and prevent errors in the analysis, the following preprocessing steps were performed:
1.	**Libraries and Data Loading**: Essential libraries (pandas, numpy, seaborn, and matplotlib) were imported, and the dataset was loaded using pd.read_csv().
2.	**Handling Missing Values**: Missing values in the "Province/State" column were filled with "Unknown." Other columns had no missing values.
3.	**Data Cleaning**: Duplicate rows were dropped, and the "Date" column was converted to datetime format.
4.	**Feature Engineering**: New columns were added:
 - **Mortality Rate**: (Deaths / Confirmed * 100)
 - **Recovery Rate**: (Recovered / Confirmed * 100)
 - **Daily Growth Rate**: Percentage change in confirmed cases.
5. **Handling Infinite Values**: Replaced infinite values with NaN and filled them with 0 for clean analysis.
   The cleaned dataset, with enhanced features, was prepared for analysis, ensuring no missing, duplicate, or invalid values.

**Import Libraries**
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

**Load and Explore dataset**
```python
df = pd.read_csv(“covid_19_clean_complete.csv”)
print(df.columns)
print(df.head())
print(df.info())
```

**Data Cleaning**
```python
print(df.isnull().sum())
df[“Province/State”].fillna(“Unknown”, inplace=True)
df.drop_duplicates(inplace=True)
df[“Date”] = pd.to_datetime(df[“Date”])
```

**Feature Engineering**
```python
df[“Mortality Rate”] = df[“Deaths”] / df[“Confirmed”] * 100
df["Recovery Rate"] = df["Recovered"] / df["Confirmed"] * 100
df[“Daily Growth Rate”] = df.groupby([“Country/Region”, “Province/State”])[“Confirmed”].pct_change()
```

**Summary Statistics**
```python
print(df.describe())
```

**Handle infinite or NaN values created during pct_change or division**
```python
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)
```

## Exploratory Data Analysis and Insights
1.	**COVID-19 Confirmed Cases Over Time**: A line plot was used to identify trends and patterns in the global spread of the virus. It was discovered that, by late March 2020, there was a significant increase in the number of confirmed cases. The virus's spread continued to escalate daily from that point.

#### Line plot visualization
```python
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x=”Date”, y=”Confirmed”)
plt.title(“COVID-19 Confirmed Cases Over Time”)
plt.xlabel("Date") 
plt.ylabel("Confirmed Cases")
plt.show()
```

2.	**Relationship Between Confirmed Cases and Deaths**: A scatter plot was used to analyze the correlation between confirmed cases and deaths. It was observed that as the number of confirmed cases increased, the number of deaths also rose.

#### Scatter plot visualization
```python
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x=”Confirmed”, y=”Deaths”)
plt.title(“Relationship between Confirmed Cases and Deaths”)
plt.xlabel(“Confirmed Cases”)
plt.ylabel(“Deaths”)
plt.show()
```

3.	**Top 10 Countries by Confirmed Cases**: A bar chart highlighted the countries with the highest total confirmed cases. The United States had the highest number, followed by Brazil, Russia, and India.

#### Total Cases by Country (Vertical Bar Chart)
```python
confirmed_cases_by_country = df.groupby(“Country/Region”)[“Confirmed”].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=confirmed_cases_by_country.index, y=confirmed_cases_by_country.values, palette="viridis")
plt.title("Top 10 Countries by Confirmed Cases")
plt.xlabel("Country")
plt.ylabel("Total Confirmed Cases")
plt.xticks(rotation=45)
plt.show()
```
These visualizations provided critical insights into the spread and severity of COVID-19, aiding in understanding and planning public health interventions.

## Time Series Predictive Model, Evaluation, and Findings
To forecast the trend of confirmed COVID-19 cases, the following steps were taken:
1.	**Data Preparation**: The daily confirmed COVID-19 cases were used, with missing values forward-filled to ensure continuity.
2.	**Modeling**: The Holt-Winters Exponential Smoothing model was applied with additive trend and seasonality (seasonal period = 7 days).
3.	**Data Splitting**: The data was split, with 80% for training and 20% for testing. Predictions were made for the test set.
4.	**Evaluation**: The model’s Root Mean Squared Error (RMSE) was 3572.35, confirming reasonable accuracy.

**Import libraries**
```python
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
```

**Load and preprocess the dataset**
```python
df = pd.read_csv(“covid_19_clean_complete.csv”)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.ffill()
time_series_data = df[['Confirmed']]
time_series_data.index = pd.to_datetime(time_series_data.index)
if time_series_data.index.duplicated().any(): time_series_data = time_series_data.groupby(time_series_data.index).mean()

time_series_data = time_series_data.asfreq('D')
time_series_data = time_series_data.ffill()
```

**Train-test split (80% training, 20% testing)**
```python
train_size = int(len(time_series_data) * 0.8)
ts_train = time_series_data[:train_size]
ts_test = time_series_data[train_size:].copy()
```

**Initialize and fit the Holt-Winters model**
```python
hw_model = ExponentialSmoothing(
    ts_train['Confirmed'], trend='add', seasonal='add', seasonal_periods=7
).fit()
```

**Forecast for the test period**
```python
ts_test.loc[:, 'Predicted'] = hw_model.forecast(len(ts_test))
```

**Evaluate model performance using RMSE**
```python
rmse = np.sqrt(mean_squared_error(ts_test['Confirmed'], ts_test['Predicted']))
print("Time-Series Model Performance:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
```

**Visualizing the actual vs predicted values**
```python
plt.figure(figsize=(12, 6))
plt.plot(ts_train['Confirmed'], label='Train')
plt.plot(ts_test['Confirmed'], label='Test', color='blue')
plt.plot(ts_test['Predicted'], label='Predicted', color='orange')
plt.title("Holt-Winters Forecasting: Actual vs Predicted")
plt.xlabel("Date")
plt.ylabel("Confirmed Cases")
plt.legend()
plt.show()
```

### Findings:
-	Training Data (Blue Line): Showed a consistent upward trend, highlighting the exponential spread of the virus.
- Testing Data (Dark Blue Line): Followed the exponential growth observed in the training data.
-	Predicted Data (Orange Line): Closely followed the trend of the test data, demonstrating the model’s ability to predict future case counts under steady conditions.

The model effectively captured trends and seasonality in confirmed cases, providing valuable predictions for future trajectories.
































