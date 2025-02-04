#TO DO: Begin creating here!

data.head()

#look at data types

dt = data.dtypes

dt

#putting variables into a list of their type

objList = []
int64List = []
int32List = []
floatList = []
datetime64List = []


for i in dt.index:
    if dt[i] == 'object':
        objList.append(i)
    if dt[i] == 'int64':
         int64List.append(i)
    if dt[i] == 'float64':
        floatList.append(i)
    if dt[i] == 'float64':
        int32List.append(i)
    if dt[i] == 'float64':
        datetime64List.append(i)

#Seeing the unique values in each variable and the most common one, and how much data is missing for categorical variables

for i in objList:
    print(i)
    print(data[i].unique())
    g= data.groupby(i)
    print(g[i].count())
    print('\nMOST COMMON = ', data[i].mode()[0])
    # print('\nMOST MEDIAN LOAN = ', g[TARGET_A].median())
    print('MISSING = ', data[i].isna().sum())
    print('\n\n')

# top Factors for accidents without unspecified

#TODO: Plot a Bar Chart

factors = data['CONTRIBUTING FACTOR VEHICLE 1'].value_counts()

#filter out unspecified

factor_filtered = factors[factors.index != 'Unspecified']

#filter the factors to only have top

factor_filtered = factor_filtered.head(3)

plt.figure(figsize=(12, 7))

#plot Bar chart top 3 factors
sns.barplot(x=factor_filtered.index, y=factor_filtered.values, palette="magma")
plt.title('Top 10 Contributing Factors to crashes', fontsize=16)
plt.xlabel('Contr. Factors', fontsize=14)
plt.ylabel('Vehicle 1 Crashes', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


#create data sets that look at one collision factor for decomposition

data_dd = data[data['CONTRIBUTING FACTOR VEHICLE 1'] == 'Driver Inattention/Distraction']
data_frw = data[data['CONTRIBUTING FACTOR VEHICLE 1'] == 'Failure to Yield Right-of-Way']
data_ftc = data[data['CONTRIBUTING FACTOR VEHICLE 1'] == 'Following Too Closely']

#Decomp of Driver Inattention/Distraction

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Count the number of crashes per day, group by CRASH DATE
daily_crashes_dd = data_dd.loc[data_dd['CRASH DATE'] > '2020-01-01']
daily_crashes_dd = daily_crashes_dd.groupby(daily_crashes_dd['CRASH DATE']).size()

# Set plot style
sns.set(style="darkgrid")

# Plot the daily crashes time series
plt.figure(figsize=(15, 6))
plt.plot(daily_crashes_dd, label='Daily crashes')
plt.title('Daily Motor Vehicle Collisions in NYC')
plt.xlabel('Day')
plt.ylabel('# of Driver Inattention/Distraction')
plt.legend()
plt.show()

# Decompose the time series
decomposition = seasonal_decompose(daily_crashes_dd, model='additive', period=365)

# Plot the decomposed components
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
decomposition.trend.plot(ax=ax1)
ax1.set_title('Trend')
decomposition.seasonal.plot(ax=ax2)
ax2.set_title('Seasonal')

# Add vertical line for Christmas day on seasonal plot
ax2.axvline(pd.to_datetime('2020-12-25'), color='red', linestyle='--', label='Christmas Day')
ax2.axvline(pd.to_datetime('2021-12-25'), color='red', linestyle='--', label='Christmas Day')
ax2.axvline(pd.to_datetime('2022-12-25'), color='red', linestyle='--', label='Christmas Day')
ax2.axvline(pd.to_datetime('2023-12-25'), color='red', linestyle='--', label='Christmas Day')

# Add vertical line for Thanksgiving day on seasonal plot
ax2.axvline(pd.to_datetime('2020-11-26'), color='green', linestyle='--', label='Thanksgiving')
ax2.axvline(pd.to_datetime('2021-11-25'), color='green', linestyle='--', label='Thanksgiving')
ax2.axvline(pd.to_datetime('2022-11-24'), color='green', linestyle='--', label='Thanksgiving')
ax2.axvline(pd.to_datetime('2023-11-23'), color='green', linestyle='--', label='Thanksgiving')

# Add vertical line for Labor day on seasonal plot
ax2.axvline(pd.to_datetime('2020-09-07'), color='yellow', linestyle='--', label='Labor Day')
ax2.axvline(pd.to_datetime('2021-09-06'), color='yellow', linestyle='--', label='Labor Day')
ax2.axvline(pd.to_datetime('2022-09-05'), color='yellow', linestyle='--', label='Labor Day')
ax2.axvline(pd.to_datetime('2023-09-04'), color='yellow', linestyle='--', label='Labor Day')

# Add vertical line for Independence Day on seasonal plot
ax2.axvline(pd.to_datetime('2020-07-04'), color='purple', linestyle='--', label='Independence Day')
ax2.axvline(pd.to_datetime('2021-07-04'), color='purple', linestyle='--', label='Independence Day')
ax2.axvline(pd.to_datetime('2022-07-04'), color='purple', linestyle='--', label='Independence Day')
ax2.axvline(pd.to_datetime('2023-07-04'), color='purple', linestyle='--', label='Independence Day')

# Add vertical line for Memorial Day on seasonal plot
ax2.axvline(pd.to_datetime('2020-05-25'), color='orange', linestyle='--', label='Memorial Day')
ax2.axvline(pd.to_datetime('2021-05-31'), color='orange', linestyle='--', label='Memorial Day')
ax2.axvline(pd.to_datetime('2022-05-30'), color='orange', linestyle='--', label='Memorial Day')
ax2.axvline(pd.to_datetime('2023-05-29'), color='orange', linestyle='--', label='Memorial Day')


decomposition.resid.plot(ax=ax3)
ax3.set_title('Residuals')
plt.tight_layout()
plt.show()

#Decomp of Failure to Yield Right-of-Way

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Count the number of crashes per day, group by CRASH DATE
daily_crashes_frw = data_frw.loc[data_frw['CRASH DATE'] > '2020-01-01']
daily_crashes_frw = daily_crashes_frw.groupby(daily_crashes_frw['CRASH DATE']).size()

# Set plot style
sns.set(style="darkgrid")

# Plot the daily crashes time series
plt.figure(figsize=(15, 6))
plt.plot(daily_crashes_frw, label='Daily crashes')
plt.title('Daily Motor Vehicle Collisions in NYC')
plt.xlabel('Day')
plt.ylabel('# of Failure to Yield Right-of-Way')
plt.legend()
plt.show()

# Decompose the time series
decomposition = seasonal_decompose(daily_crashes_frw, model='additive', period=365)

# Plot the decomposed components
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
decomposition.trend.plot(ax=ax1)
ax1.set_title('Trend')
decomposition.seasonal.plot(ax=ax2)
ax2.set_title('Seasonal')

# Add vertical line for Christmas day on seasonal plot
ax2.axvline(pd.to_datetime('2020-12-25'), color='red', linestyle='--', label='Christmas Day')
ax2.axvline(pd.to_datetime('2021-12-25'), color='red', linestyle='--', label='Christmas Day')
ax2.axvline(pd.to_datetime('2022-12-25'), color='red', linestyle='--', label='Christmas Day')
ax2.axvline(pd.to_datetime('2023-12-25'), color='red', linestyle='--', label='Christmas Day')

# Add vertical line for Thanksgiving day on seasonal plot
ax2.axvline(pd.to_datetime('2020-11-26'), color='green', linestyle='--', label='Thanksgiving')
ax2.axvline(pd.to_datetime('2021-11-25'), color='green', linestyle='--', label='Thanksgiving')
ax2.axvline(pd.to_datetime('2022-11-24'), color='green', linestyle='--', label='Thanksgiving')
ax2.axvline(pd.to_datetime('2023-11-23'), color='green', linestyle='--', label='Thanksgiving')

# Add vertical line for Labor day on seasonal plot
ax2.axvline(pd.to_datetime('2020-09-07'), color='yellow', linestyle='--', label='Labor Day')
ax2.axvline(pd.to_datetime('2021-09-06'), color='yellow', linestyle='--', label='Labor Day')
ax2.axvline(pd.to_datetime('2022-09-05'), color='yellow', linestyle='--', label='Labor Day')
ax2.axvline(pd.to_datetime('2023-09-04'), color='yellow', linestyle='--', label='Labor Day')

# Add vertical line for Independence Day on seasonal plot
ax2.axvline(pd.to_datetime('2020-07-04'), color='purple', linestyle='--', label='Independence Day')
ax2.axvline(pd.to_datetime('2021-07-04'), color='purple', linestyle='--', label='Independence Day')
ax2.axvline(pd.to_datetime('2022-07-04'), color='purple', linestyle='--', label='Independence Day')
ax2.axvline(pd.to_datetime('2023-07-04'), color='purple', linestyle='--', label='Independence Day')

# Add vertical line for Memorial Day on seasonal plot
ax2.axvline(pd.to_datetime('2020-05-25'), color='orange', linestyle='--', label='Memorial Day')
ax2.axvline(pd.to_datetime('2021-05-31'), color='orange', linestyle='--', label='Memorial Day')
ax2.axvline(pd.to_datetime('2022-05-30'), color='orange', linestyle='--', label='Memorial Day')
ax2.axvline(pd.to_datetime('2023-05-29'), color='orange', linestyle='--', label='Memorial Day')


decomposition.resid.plot(ax=ax3)
ax3.set_title('Residuals')
plt.tight_layout()
plt.show()

#Decomp of Following Too Closely

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Count the number of crashes per day, group by CRASH DATE
daily_crashes_ftc = data_ftc.loc[data_ftc['CRASH DATE'] > '2020-01-01']
daily_crashes_ftc = daily_crashes_ftc.groupby(daily_crashes_ftc['CRASH DATE']).size()

# Set plot style
sns.set(style="darkgrid")

# Plot the daily crashes time series
plt.figure(figsize=(15, 6))
plt.plot(daily_crashes_ftc, label='Daily crashes')
plt.title('Daily Motor Vehicle Collisions in NYC')
plt.xlabel('Day')
plt.ylabel('# of Following Too Closely')
plt.legend()
plt.show()

# Decompose the time series
decomposition = seasonal_decompose(daily_crashes_ftc, model='additive', period=365)

# Plot the decomposed components
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
decomposition.trend.plot(ax=ax1)
ax1.set_title('Trend')
decomposition.seasonal.plot(ax=ax2)
ax2.set_title('Seasonal')

# Add vertical line for Christmas day on seasonal plot
ax2.axvline(pd.to_datetime('2020-12-25'), color='red', linestyle='--', label='Christmas Day')
ax2.axvline(pd.to_datetime('2021-12-25'), color='red', linestyle='--', label='Christmas Day')
ax2.axvline(pd.to_datetime('2022-12-25'), color='red', linestyle='--', label='Christmas Day')
ax2.axvline(pd.to_datetime('2023-12-25'), color='red', linestyle='--', label='Christmas Day')

# Add vertical line for Thanksgiving day on seasonal plot
ax2.axvline(pd.to_datetime('2020-11-26'), color='green', linestyle='--', label='Thanksgiving')
ax2.axvline(pd.to_datetime('2021-11-25'), color='green', linestyle='--', label='Thanksgiving')
ax2.axvline(pd.to_datetime('2022-11-24'), color='green', linestyle='--', label='Thanksgiving')
ax2.axvline(pd.to_datetime('2023-11-23'), color='green', linestyle='--', label='Thanksgiving')

# Add vertical line for Labor day on seasonal plot
ax2.axvline(pd.to_datetime('2020-09-07'), color='yellow', linestyle='--', label='Labor Day')
ax2.axvline(pd.to_datetime('2021-09-06'), color='yellow', linestyle='--', label='Labor Day')
ax2.axvline(pd.to_datetime('2022-09-05'), color='yellow', linestyle='--', label='Labor Day')
ax2.axvline(pd.to_datetime('2023-09-04'), color='yellow', linestyle='--', label='Labor Day')

# Add vertical line for Independence Day on seasonal plot
ax2.axvline(pd.to_datetime('2020-07-04'), color='purple', linestyle='--', label='Independence Day')
ax2.axvline(pd.to_datetime('2021-07-04'), color='purple', linestyle='--', label='Independence Day')
ax2.axvline(pd.to_datetime('2022-07-04'), color='purple', linestyle='--', label='Independence Day')
ax2.axvline(pd.to_datetime('2023-07-04'), color='purple', linestyle='--', label='Independence Day')

# Add vertical line for Memorial Day on seasonal plot
ax2.axvline(pd.to_datetime('2020-05-25'), color='orange', linestyle='--', label='Memorial Day')
ax2.axvline(pd.to_datetime('2021-05-31'), color='orange', linestyle='--', label='Memorial Day')
ax2.axvline(pd.to_datetime('2022-05-30'), color='orange', linestyle='--', label='Memorial Day')
ax2.axvline(pd.to_datetime('2023-05-29'), color='orange', linestyle='--', label='Memorial Day')

decomposition.resid.plot(ax=ax3)
ax3.set_title('Residuals')
plt.tight_layout()
plt.show()

#TODO: Create a heatmap leveraging the latitude and longitude variables to determine where the most crashes are occurring
from folium.plugins import HeatMap

#create data heat map for just vehicle 1 and collision factors on holidays
data_heat = data[data['CONTRIBUTING FACTOR VEHICLE 1'].isin(['Driver Inattention/Distraction','Failure to Yield Right-of-Way', 'Following Too Closely'])
& data['CRASH DATE'].isin(['2020-12-25','2021-12-25','2022-12-25','2023-12-25','2020-11-26','2021-11-25','2022-11-24',
                             '2023-11-23','2020-09-07','2021-09-06','2022-07-04','2023-09-04','2020-07-04','2021-07-04','2022-07-04',
                             '2023-07-04','2020-05-25','2021-05-31','2022-05-30','2023-05-29'])]

# Drop rows with missing latitude and longitude values
data_geo = data_heat.dropna(subset=['LATITUDE', 'LONGITUDE'])

# Create a base map
m = folium.Map(location=[40.730610, -73.935242], zoom_start=10)  # Centered around NYC

# Create a heatmap
heat_data = [[row['LATITUDE'], row['LONGITUDE']] for index, row in data_geo.iterrows()]
HeatMap(heat_data, radius=8, max_zoom=13).add_to(m)

m.save("Heatmap_5.html")


#TODO: Continue building your heatmap
# Sample a subset of the data for visualization
sample_data_severity = data_geo

# Create a base map
m_severity = folium.Map(location=[40.730610, -73.935242], zoom_start=10)

# Add crashes to the map with color coding and shape coding based on type
for index, row in sample_data_severity.iterrows():
    if row['CONTRIBUTING FACTOR VEHICLE 1'] == 'Driver Inattention/Distraction':
        color = "RED"  # Fatalities

        folium.features.RegularPolygonMarker(
          location=[row['LATITUDE'], row['LONGITUDE']],
          number_of_sides=3,
          radius=5,
          gradient = False,
          color=color,
          fill=True,
          fill_color=color
        ).add_to(m_severity)


    elif row['CONTRIBUTING FACTOR VEHICLE 1'] == 'Failure to Yield Right-of-Way':
        color = "PURPLE"  # Injuries
        folium.CircleMarker(
          location=[row['LATITUDE'], row['LONGITUDE']],
          radius=5,
          color=color,
          fill=True,
          fill_color=color
       ).add_to(m_severity)
    else:
        color = "GREY"  # No injuries or fatalities
        folium.features.RegularPolygonMarker(
          location=[row['LATITUDE'], row['LONGITUDE']],
          number_of_sides=4,
          radius=5,
          gradient = False,
          color=color,
          fill=True,
          fill_color=color
        ).add_to(m_severity)


m_severity.save("severity_5.html")

#create pie chart of top 3 collision facotrs

x = data_geo['CONTRIBUTING FACTOR VEHICLE 1'].value_counts(dropna=False)
theLabels = x.axes[0].tolist()
theSlices = list(x)

explodeList = [0 for i in theSlices]
explodeList[0] = 0.07
explodeList[1] = 0.05
explodeList[2] = 0.03

plt.pie(theSlices, labels = theLabels, startangle = 90, explode = explodeList,  shadow = True, autopct = '%1.1f%%')
plt.title('Pie Chart: CONTRIBUTING FACTOR VEHICLE 1')
plt.show()