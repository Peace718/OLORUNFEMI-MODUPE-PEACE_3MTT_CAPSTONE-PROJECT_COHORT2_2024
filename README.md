# OLORUNFEMI-MODUPE-PEACE_3MTT_CAPSTONE-PROJECT_COHORT2_2024

# PROJECT TOPIC: Predictive Modeling for COVID-19 in Public Health

## Table of Contents
- [Introduction](#Introduction)
- [Project Overview](#project-overview)
- [Data Design](#data-design)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis and Insights](#exploratory-data-analysis-and-insights)
- [Time Series Predictive Model, Evaluation, and Findings](#time-series-predictive-model-evaluation-and-findings)
- [Classification Predictive Model, Evaluation, and Findings](#classification-predictive-model-evaluation-and-findings)
- [Data Visualization](#data-visualization)
- [Recommendations](#recommendations)
- [Conclusion](#conclusion)

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
-	**Training Data (Blue Line)**: Showed a consistent upward trend, highlighting the exponential spread of the virus.
- **Testing Data (Dark Blue Line)**: Followed the exponential growth observed in the training data.
-	**Predicted Data (Orange Line)**: Closely followed the trend of the test data, demonstrating the model’s ability to predict future case counts under steady conditions.

The model effectively captured trends and seasonality in confirmed cases, providing valuable predictions for future trajectories.

## Classification Predictive Model, Evaluation, and Findings
A Random Forest Classifier was used to classify whether the number of confirmed COVID-19 cases exceeded 1,000 based on daily growth rate, mortality rate, and recovered cases. Key steps included:
1.	**Feature Engineering**: Derived features like daily growth rate and mortality rate.
2.	**Modeling**: An 80/20 train-test split was used, with features including daily growth rate, mortality rate, and recovered cases.
3.	**Evaluation Metrics**:
-	Accuracy: 98%
-	Precision: 99%
-	Recall: 96%
-	F1-Score: 96%

**Import libraries**
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
```

**Load the dataset**
```python
df = pd.read_csv(“covid_19_clean_complete.csv”)
```

**Ensure no missing values in the dataset**
```python
df = df.ffill()
```

**Create missing columns**
```python
df["daily_growth_rate"] = df["Confirmed"].pct_change().fillna(0) * 100
df["mortality_rate"] = (df["Deaths"] / df["Confirmed"]).fillna(0) * 100
if "Recovered" in df.columns: df.rename(columns={"Recovered": "recovered"}, inplace=True)
```

**Define features and target**
```python

features = ["daily_growth_rate", "mortality_rate", "recovered"]
df["target"] = (df["Confirmed"] > 1000).astype(int)        # Example binary classification
X = df[features]
y = df["target"]
```

**Replace infinity and handle NaN**
```python
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())                                           
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

**Split data into train and test sets**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Initialize and train the Random Forest Classifier**
```python
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
```

**Make predictions**
```python
y_pred = rf_model.predict(X_test)
```

**Evaluate model performance**
```python
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
```

**Display evaluation metrics**
```python
print("Classification Model Performance:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
```

**Print detailed classification report**
```python
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))
```

### Findings:
The Random Forest model demonstrated robust performance, with accuracy (98%) and precision (99%) making it a valuable tool for monitoring trends and threshold cases. This model effectively classified COVID-19 cases and provided reliable predictions for public health applications.

## Data Visualization
Please view the visualization in the [Google Slides presentation here](https://docs.google.com/presentation/d/1KKoaHO9lGhAuc0K4Il3C9xWB4C6wyKta4C2cRmOb7Ao/edit?usp=sharing)

## Recommendations
Based on the analysis, the following recommendations are proposed:
1.	**Strengthen Surveillance Systems**: Enhance real-time data collection and monitoring to improve the accuracy of predictive models.
2.	**Resource Allocation**: Use insights from predictive models to prioritize resources for heavily impacted regions.
3.	**Policy Adjustments**: Tailor public health policies based on trends in mortality and recovery rates to optimize health outcomes.
4.	**Continuous Model Refinement**: Update the models with newer data to maintain relevance and improve accuracy.
5.	**Public Awareness Campaigns**: Leverage insights to design campaigns focused on mitigating virus spread in regions with high growth rates.

## Conclusion
This project demonstrates the power of data-driven approaches in public health decision-making. By leveraging predictive models and data analysis, we provided actionable insights into COVID-19’s spread and severity. The Holt-Winters model effectively captured time-series trends, while the Random Forest Classifier showcased strong performance in case classification. These models offer valuable tools for future outbreak management and highlight the importance of integrating data science into public health strategies. Continuous data updates and model refinements are essential to address evolving challenges and improve health outcomes globally.
































