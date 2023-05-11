# -*- coding: utf-8 -*-
"""
Created on Tue May  9 17:05:30 2023

@author: Hp
"""

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import seaborn as sns
import cluster_tools as ct
import scipy.optimize as opt
from scipy.optimize import curve_fit


#Using Def function to read the file

"""
Creating a def function to read in our datasets and skiprows the first 4 rows
"""


def read_data(filename, **others):
    """
    A function that reads in world bank indicator  data and returns the skip the first 4 rows
        filename: the name of the world bank data that will be read for analysis 
        and manupulation

        **others: other arguments to pass into the functions as need be, such
        as skipping the first 4 rows

    Returns: 
        The  dataset that has been read in with its first 4 rows skipped
    """

    # Reading in the climate dataset for to be used for analysis with first 4 rows skipped
    world_data = pd.read_csv(filename, skiprows=4)

    return world_data


# Define the filenames
laborforce = 'API_SL.TLF.TOTL.IN_DS2_en_csv_v2_5359352.csv'
gdppercapital = 'API_NY.GDP.PCAP.KD.ZG_DS2_en_csv_v2_5358515.csv'

# Read in the data using pandas, skipping the first 4 rows
labor = pd.read_csv(laborforce, skiprows=4)
gdp = pd.read_csv(gdppercapital, skiprows=4)

#checking descriptive statistics
labor.describe()
gdp.describe()

# Selecting the countries needed
labor = labor[labor['Country Name'].isin(
    ['Ghana', 'China', 'United States', 'United Kingdom', 'Belgium'])]

#Dropping the column i will not be using
labor = labor.drop(
    ['Indicator Name', 'Country Code', 'Indicator Code'], axis=1)

# Reset the index
labor.reset_index(drop=True, inplace=True)
labor.loc[:, :]


#Years chosen for the clustering analysis
labor_ex = labor[['Country Name', '1991', '2021']]
labor_ex.describe()

#checking for missing values
labor_ex.isna().sum()


#transposing the data
labor_t = labor_ex.T
labor_t.columns = labor_t.iloc[0]
labor_t = labor_t.iloc[1:]
labor_t.describe()
labor_t = labor_t.apply(pd.to_numeric)

# taking the years i will be working with from the dataset
labor_year = labor_ex[['1991', '2021']]
labor_year.describe()

#dropping missing values
labor_year.dropna(inplace=True)

#checking for correlation between the years choosen
#correlation
corr = labor_year.corr()

#heatmap
ct.map_corr(labor_year)

#scatter plot
pd.plotting.scatter_matrix(labor_year, figsize=(9.0, 9.0))
plt.tight_layout()  # helps to avoid overlap of labels
plt.show()

labor_cluster = labor_year[['1991', '2021']].copy()

#Normalizing the data and storing minimum and maximum value
labor_norm, labor_min, labor_max = ct.scaler(labor_cluster)
print(labor_norm.describe())

#caculating the best clustering number
print('n score')
#loop over trail numbers of clusters calculaing the silhouette
for i in range(2, 5):
    #set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=i)
    kmeans.fit(labor_cluster)

    #extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print(i, skmet.silhouette_score(labor_cluster, labels))
#2 and 3 has the highest silhoutte score respectively
#plotting for 2 clusters

nclusters = 3  # number of cluster centres

kmeans = cluster.KMeans(n_clusters=nclusters)
kmeans.fit(labor_norm)

#extract labels and cluster centres
labels = kmeans.labels_

#extracting the estimated number of clusster
cen = kmeans.cluster_centers_

plt.figure(figsize=(6.0, 6.0))
# scatter plot with colours selected using the cluster numbers
plt.scatter(labor_norm["1991"], labor_norm["2021"], c=labels, cmap="tab10")
# colour map Accent selected to increase contrast between colours

# show cluster centres
xcen = cen[:, 0]
ycen = cen[:, 1]
plt.scatter(xcen, ycen, c="k", marker="d", s=80)
# c = colour, s = size

plt.xlabel("labor(1991)")
plt.ylabel("labor(2021)")
plt.title("3 clusters")
plt.show()

#Scaling back to the original data and creating a plot on it original scale

plt.figure(figsize=(6.0, 6.0))

# now using the original dataframe
sns.scatterplot(x='1991', y='2021', data=labor,
                hue='Country Name', c=labels, cmap="tab10")
# colour map Accent selected to increase contrast between colours

# rescale and show cluster centres
scen = ct.backscale(cen, labor_min, labor_max)
xc = scen[:, 0]
yc = scen[:, 1]
plt.scatter(xc, yc, c="k", marker="d", s=80)

plt.xlabel("1991")
plt.ylabel("2021")
plt.title("3 clusters")
plt.show()

# Curve fitting

# calling in the urban population data
print(labor)

# transposing the data
labor = labor.T

#dropping missing values
labor.dropna(inplace=True)

# Making the country name the colums
labor.columns = labor.iloc[0]

# Selecting only the years
labor = labor.iloc[1:]

#converting the columns and the index
labor = labor.apply(pd.to_numeric)
labor.index = pd.to_numeric(labor.index)
labor.reset_index(inplace=True)

labor.rename(columns={'index': 'Year'}, inplace=True)
print(labor)

df_fitG = labor[['Year', 'Ghana']]

plt.plot(labor['Year'], labor['Ghana'])

df_fitG.T
plt.plot(labor['Year'], labor['Ghana'])


# Using polynomial function for curve fitting and forecasting the labor force for Ghana

def polynomial(t, *coefficients):
    """
    Computes a polynomial function
    
    Parameters:
        t: The time
        coefficients: Coefficients of the polynomial
    
    Returns:
        The total labor force at the given time
    """
    return np.polyval(coefficients, t)


# Data for Ghana
years_ghana = labor['Year'].values
total_laborforce_ghana = labor['Ghana'].values

# Degree of the polynomial
degree = 6

# Fitting the polynomial model for Ghana
coefficients_ghana = np.polyfit(years_ghana, total_laborforce_ghana, degree)

# Predictions for 2030 and 2040 for Ghana
prediction_2030_ghana = polynomial(2030, *coefficients_ghana)
prediction_2040_ghana = polynomial(2040, *coefficients_ghana)

print("Total labor force prediction for Ghana in 2030:", prediction_2030_ghana)
print("Total labor force prediction for Ghana in 2040:", prediction_2040_ghana)

# Generating points for the fitted curve
curve_years_ghana = np.linspace(min(years_ghana), max(years_ghana), 100)
curve_laborforce_ghana = polynomial(curve_years_ghana, *coefficients_ghana)

# Error range
residuals_ghana = total_laborforce_ghana - \
    polynomial(years_ghana, *coefficients_ghana)
sigma_ghana = np.std(residuals_ghana)
lower_ghana = curve_laborforce_ghana - sigma_ghana
upper_ghana = curve_laborforce_ghana + sigma_ghana

# Plotting the data and fitted curve for Ghana
plt.figure(figsize=(10, 6))
plt.plot(years_ghana, total_laborforce_ghana, 'ro', label='Data')
plt.plot(curve_years_ghana, curve_laborforce_ghana, 'b-', label='Fitted Curve')
plt.plot(2030, prediction_2030_ghana, 'go',
         markersize=10, label='Prediction for 2030')
plt.plot(2040, prediction_2040_ghana, 'yo',
         markersize=10, label='Prediction for 2040')
plt.fill_between(curve_years_ghana, lower_ghana, upper_ghana,
                 alpha=0.2, color='blue', label='Confidence Range')
plt.xlabel('Year')
plt.ylabel('Total Labor Force')
plt.title('Polynomial Fit for Total Labor Force of Ghana')
plt.legend()
plt.grid(True)
plt.show()


# Using polynomial function for curve fitting and forecasting the labor force for China

def polynomial(t, *coefficients):
    """
    Computes a polynomial function
    
    Parameters:
        t: The time
        coefficients: Coefficients of the polynomial
    
    Returns:
        The total labor force at the given time
    """
    return np.polyval(coefficients, t)

# Data for China


years_china = labor['Year'].values
total_laborforce_china = labor['China'].values

# Degree of the polynomial
degree = 6

# Fitting the polynomial model for China
coefficients_china = np.polyfit(years_ghana, total_laborforce_china, degree)

# Predictions for 2030 and 2040 for China
prediction_2030_china = polynomial(2030, *coefficients_china)
prediction_2040_china = polynomial(2040, *coefficients_china)

print("Total labor force prediction for China in 2030:", prediction_2030_china)
print("Total labor force prediction for China in 2040:", prediction_2040_china)

# Generating points for the fitted curve
curve_years_china = np.linspace(min(years_china), max(years_china), 100)
curve_laborforce_china = polynomial(curve_years_china, *coefficients_china)

# Error range
residuals_china = total_laborforce_china - \
    polynomial(years_china, *coefficients_china)
sigma_china = np.std(residuals_china)
lower_china = curve_laborforce_china - sigma_china
upper_china = curve_laborforce_china + sigma_china

# Plotting the data and fitted curve for China
plt.figure(figsize=(10, 6))
plt.plot(years_china, total_laborforce_china, 'ro', label='Data')
plt.plot(curve_years_china, curve_laborforce_china, 'b-', label='Fitted Curve')
plt.plot(2030, prediction_2030_china, 'go',
         markersize=10, label='Prediction for 2030')
plt.plot(2040, prediction_2040_china, 'yo',
         markersize=10, label='Prediction for 2040')
plt.fill_between(curve_years_china, lower_china, upper_china,
                 alpha=0.2, color='blue', label='Confidence Range')
plt.xlabel('Year')
plt.ylabel('Total Labor Force')
plt.title('Polynomial Fit for Total Labor Force of China')
plt.legend()
plt.grid(True)
plt.show()


# Working on the data of GDP per capita growth (annual %)

# Selecting the countries needed
gdp = gdp[gdp['Country Name'].isin(
    ['Ghana', 'China', 'United States', 'United Kingdom', 'Belgium'])]

#Dropping the column i will not be using
gdp = gdp.drop(['Indicator Name', 'Country Code', 'Indicator Code'], axis=1)

# Reset the index
gdp.reset_index(drop=True, inplace=True)
gdp.loc[:, :]

print(gdp)

# transposing the data
gdp = gdp.T

#dropping missing values
gdp.dropna(inplace=True)

# Making the country name the colums
gdp.columns = gdp.iloc[0]

# Selecting only the years
gdp = gdp.iloc[1:]

#converting the columns and the index
gdp = gdp.apply(pd.to_numeric)
gdp.index = pd.to_numeric(gdp.index)
gdp.reset_index(inplace=True)

gdp.rename(columns={'index': 'Year'}, inplace=True)
print(gdp)

df_fitG = gdp[['Year', 'Ghana']]

plt.plot(gdp['Year'], gdp['Ghana'])

df_fitG.T
plt.plot(gdp['Year'], gdp['Ghana'])


# Using polynomial function for curve fitting and forecasting the GDP per capita growth (annual %) for Ghana

def polynomial(t, *coefficients):
    """
    Computes a polynomial function
    
    Parameters:
        t: The time
        coefficients: Coefficients of the polynomial
    
    Returns:
        The total labor force at the given time
    """
    return np.polyval(coefficients, t)


# Data for Ghana
years_gdp_ghana = gdp['Year'].values
total_gdp_ghana = gdp['Ghana'].values

# Degree of the polynomial
degree = 6

# Fitting the polynomial model for Ghana
coefficients_gdp_ghana = np.polyfit(years_gdp_ghana, total_gdp_ghana, degree)

# Predictions for 2030 and 2040 for Ghana
prediction_2030_gdp_ghana = polynomial(2030, *coefficients_gdp_ghana)
prediction_2040_gdp_ghana = polynomial(2040, *coefficients_gdp_ghana)

print("GDP per capita growth (annual %) for Ghana in 2030:",
      prediction_2030_gdp_ghana)
print("GDP per capita growth (annual %) for Ghana in 2040:",
      prediction_2040_gdp_ghana)

# Generating points for the fitted curve
curve_years_gdp_ghana = np.linspace(
    min(years_gdp_ghana), max(years_gdp_ghana), 100)
curve_laborforce_gdp_ghana = polynomial(
    curve_years_gdp_ghana, *coefficients_gdp_ghana)

# Error range
residuals_gdp_ghana = total_gdp_ghana - \
    polynomial(years_gdp_ghana, *coefficients_gdp_ghana)
sigma_gdp_ghana = np.std(residuals_gdp_ghana)
lower_gdp_ghana = curve_laborforce_gdp_ghana - sigma_gdp_ghana
upper_gdp_ghana = curve_laborforce_gdp_ghana + sigma_gdp_ghana

# Plotting the data and fitted curve for Ghana
plt.figure(figsize=(10, 6))
plt.plot(years_gdp_ghana, total_gdp_ghana, 'ro', label='Data')
plt.plot(curve_years_gdp_ghana, curve_laborforce_gdp_ghana,
         'b-', label='Fitted Curve')
plt.plot(2030, prediction_2030_gdp_ghana, 'go',
         markersize=10, label='Prediction for 2030')
plt.plot(2040, prediction_2040_gdp_ghana, 'yo',
         markersize=10, label='Prediction for 2040')
plt.fill_between(curve_years_gdp_ghana, lower_gdp_ghana,
                 upper_gdp_ghana, alpha=0.2, color='blue', label='Confidence Range')
plt.xlabel('Year')
plt.ylabel('GDP per capital growth')
plt.title('Polynomial Fit for GDP per capita growth (annual %) for Ghana')
plt.legend()
plt.grid(True)
plt.show()


# Using polynomial function for curve fitting and forecasting the GDP per capita growth (annual %) for China

def polynomial(t, *coefficients):
    """
    Computes a polynomial function
    
    Parameters:
        t: The time
        coefficients: Coefficients of the polynomial
    
    Returns:
        The total labor force at the given time
    """
    return np.polyval(coefficients, t)


# Data for china
years_gdp_china = gdp['Year'].values
total_gdp_china = gdp['China'].values

# Degree of the polynomial
degree = 6

# Fitting the polynomial model for China
coefficients_gdp_china = np.polyfit(years_gdp_china, total_gdp_china, degree)

# Predictions for 2030 and 2040 for China
prediction_2030_gdp_china = polynomial(2030, *coefficients_gdp_china)
prediction_2040_gdp_china = polynomial(2040, *coefficients_gdp_china)

print("GDP per capita growth (annual %) for China in 2030:",
      prediction_2030_gdp_china)
print("GDP per capita growth (annual %) for China in 2040:",
      prediction_2040_gdp_china)

# Generating points for the fitted curve
curve_years_gdp_china = np.linspace(
    min(years_gdp_china), max(years_gdp_china), 100)
curve_laborforce_gdp_china = polynomial(
    curve_years_gdp_china, *coefficients_gdp_china)

# Error range
residuals_gdp_china = total_gdp_china - \
    polynomial(years_gdp_china, *coefficients_gdp_china)
sigma_gdp_china = np.std(residuals_gdp_china)
lower_gdp_china = curve_laborforce_gdp_china - sigma_gdp_china
upper_gdp_china = curve_laborforce_gdp_china + sigma_gdp_china

# Plotting the data and fitted curve for China
plt.figure(figsize=(10, 6))
plt.plot(years_gdp_china, total_gdp_china, 'ro', label='Data')
plt.plot(curve_years_gdp_china, curve_laborforce_gdp_china,
         'b-', label='Fitted Curve')
plt.plot(2030, prediction_2030_gdp_china, 'go',
         markersize=10, label='Prediction for 2030')
plt.plot(2040, prediction_2040_gdp_china, 'yo',
         markersize=10, label='Prediction for 2040')
plt.fill_between(curve_years_gdp_china, lower_gdp_china,
                 upper_gdp_china, alpha=0.2, color='blue', label='Confidence Range')
plt.xlabel('Year')
plt.ylabel('GDP per capital growth')
plt.title('Polynomial Fit for GDP per capita growth (annual %) for China')
plt.legend()
plt.grid(True)
plt.show()
