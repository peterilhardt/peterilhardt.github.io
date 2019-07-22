---
layout: post
title: What Makes a Country Wealthy? Delving into UN Data
---

Article 1 of the [Charter of the United Nations](https://www.un.org/en/sections/un-charter/chapter-i/index.html) outlines the purposes of the United Nations as a whole, of which one item reads:

> To achieve international co-operation in solving international problems of an economic, social, cultural, or humanitarian character, and in promoting and encouraging respect for human rights and for fundamental freedoms for all without distinction as to race, sex, language, or religion.

Undoubtedly one crucial aspect of this is helping developing nations achieve satisfactory economic health, which is often characterized by a broad metric such as Gross Domestic Product (GDP). But what ultimately makes a country wealthy? Are there simple economic indicators such as unemployment rate that generally capture underlying differences in economic health, or are there other non-economic factors that can shed light on the subtle intricacies of such complex systems? 

To tackle this question, I turned to the *United Nations Statistics Division* ([UNSD](https://unstats.un.org/home/)), which maintains a large repository of data on various economic, social, demographic, and other metrics of the world's officially recognized countries over many years: [UNdata](http://data.un.org/). 

I set out to build regression models that would predict GDP per capita (a more accurate representation of a nation's wealth than raw GDP) using many of these available features. To start, I downloaded the publicly-available .csv files and imported the data into Jupyter Notebooks, using a script such as:

```python
import numpy as np
import pandas as pd

def read_un_data(filepath, df):   
    df = pd.read_csv(filepath, encoding = 'latin-1', skiprows = 1)
    #drop unnecessary columns
    df = df.drop(columns = ['Footnotes', 'Source', 'Region/Country/Area'])\
        .rename(columns = {'Unnamed: 1': 'Country'})
    #filter for 2005, 2010, 2015 and remove unnecessary rows
    df = df[df['Year'].isin([2005, 2010, 2015]) & ~df['Country'].isin(row_exclude)]   #row_exclude had a long list of rows I was not interested in
    return df

#Example for GDP dataset
gdp = pd.DataFrame()
gdp = read_un_data('data/SYB62_T13_201904_GDP and GDP Per Capita.csv', gdp)
gdp = gdp[gdp.Series == 'GDP per capita (US dollars)']\
    .pivot_table(index = ['Country', 'Year'], columns='Series', values = 'Value', aggfunc = 'max')\
    .reset_index().rename(columns = {'GDP per capita (US dollars)': 'gdp'})
gdp['gdp'] = gdp['gdp'].str.replace(',', '').astype(int)
```

I decided to focus on the years 2005, 2010, and 2015, and repeated a similar process for 18 of the datasets available. These were initially filtered for only rows containing countries (as opposed to aggregate metrics such as continents and regions), pivoted to display the years in one column (long format), and renamed as necessary. I also decided to scrape data on greenhouse gas emissions per country per capita in the years 2005, 2010, and 2013 from a Wikipedia table [here](https://en.wikipedia.org/wiki/List_of_countries_by_greenhouse_gas_emissions_per_capita). After merging the UN data into a single dataframe, I acquired the Wikipedia table using BeautifulSoup:

```python
import requests
from bs4 import BeautifulSoup

url = 'https://en.wikipedia.org/wiki/List_of_countries_by_greenhouse_gas_emissions_per_capita'
response = requests.get(url)
page = response.text
soup = BeautifulSoup(page, 'html5')

table = soup.find('table', {'class': 'wikitable sortable'})   #get Wikipedia table item labeled 'wikitable sortable'

links = table.find_all('a')   #the first column contains the country names as embedded links
countries = []
for country in links:
    countries.append(country.get('title'))
ghg = pd.DataFrame()
ghg['Country'] = countries[1:]

rows = table.find_all('tr')   #the other columns just contain numbers as text
rows = [row for row in rows if ('World' not in row.text) and ('Country' not in row.text)]   #remove row 'World' and the header row
year_2005 = []
year_2010 = []
year_2013 = []

for i in range(0, len(rows)):
    columns = rows[i].find_all('td')    #find all cells for each row
    year_2005.append(columns[4].get_text().strip())   #extract the cell in the fifth column (2005)
    year_2010.append(columns[5].get_text().strip())
    year_2013.append(columns[6].get_text().strip())

year_2005 = [np.nan if x == '' else x for x in year_2005]   #replace blank cells with NaN
year_2010 = [np.nan if x == '' else x for x in year_2010]
year_2013 = [np.nan if x == '' else x for x in year_2013]

ghg['2005'] = year_2005
ghg['2010'] = year_2010
ghg['2015'] = year_2013   #treating 2013 as 2015 here for compatibility with the UN data

ghg = ghg.melt(id_vars = ['Country'], value_vars = ['2005', '2010', '2015'],    #make long format
               var_name = 'Year', value_name = 'ghg_emissions')
ghg[['Year', 'ghg_emissions']] = ghg[['Year', 'ghg_emissions']].transform({'Year': int, 'ghg_emissions': float})
```

After merging this data with the UN dataframe, I was ready to begin the cleaning process. Cleaning consisted of the following steps:

* Correcting country name differences between Wikipedia and UNdata
* Removing unnecessary columns and renaming others as needed
* Fixing data types (including removing punctuation from numbers)
* Removing countries that did not have GDP per capita listed or had many missing features (>5; typically islands and small territories)
* Removing features that had many missing observations (>125)

After this, I was still left with quite a few NaN's in the data that I needed to impute. I decided to take a two-step approach for this:

* For countries/fields with only one or two missing values (i.e. missing data for one or two years but at least one year present), I filled the missing entries with the mean value of that country/field for the other years:

```python
for col in UN.columns[2:]:
    UN[col] = UN.groupby('Country')[col].transform(lambda x: x.fillna(x.mean()))
```

* For countries/fields with all three values missing (i.e. all three years), I implemented kNN imputation using the `KNN` function from the *fancyimpute* module. I decided this was more appropriate than a linear imputation approach since countries that share similar characteristics are more likely to have values for a missing field that closely resemble the actual missing values. I selected k = 9 to ensure that the three most similar countries were considered. Note that `KNN` requires a matrix rather than a dataframe:

```python
from fancyimpute import KNN

UN_num = UN.select_dtypes(include=[np.float])   #get numeric data from dataframe as matrix
UN2 = pd.DataFrame(data = KNN(9).fit_transform(UN_num), 
                   columns = UN_num.columns, index = UN_num.index)
```

Finally, for the features that weren't already normalized by population size, I normalized them accordingly. 


### Exploratory Data Analysis

This left me with a dataframe of 182 countries over three years (546 total observations) and 28 features. The features included (for more information see the UNdata webpage):

* GDP per capita
* Population density
* Total population
* Gender ratio
* Difference between total exports and imports
* Number of students enrolled in primary education
* Gross Value Added (GVA) by agriculture
* Gross Value Added (GVA) by industry
* Gross Value Added (GVA) by service
* Labor force participation rate
* Unemployment rate
* Energy production
* Energy supplied per capita
* Percentage of population with internet access
* Tourism expenditure
* Percentage of GDP spent on healthcare
* Percentage of parliament comprised of women
* Food Production Index
* Consumer Price Index
* Infant mortality rate
* Maternal mortality rate
* Life expectancy
* Year-to-year population gain (%)
* Fertility rate (number of children per woman)
* Percentage of population comprised of international migrants
* Population of concern to UNHCR (UN High Commissioner for Refugees)
* Percentage of population in urban settings
* Greenhouse gas (GHG) emissions per capita

Naturally with this many related features, there were quite a few high-magnitude correlation coefficients, as demonstrated by this correlation matrix heatmap:

![correlation_matrix_heatmap]({{ site.url }}/images/correlation_matrix_heatmap.png)

Also, the majority of the features were highly right-skewed in their distributions (as is often expected with census data), which required most features to be log- or square-root-transformed. A subset of these features and their histograms are shown in the following scatterplot matrix:

![pairplot]({{ site.url }}/images/pairplot.png)

A comparison of the 10 highest-GDP countries (averaged over the three years in the dataset) to the 10 lowest adequately captures the disparity in economic wealth observed between rich and poor nations. The highest mean GDP per capita (Luxembourg) is about $94,632 (U.S. dollars) more than the lowest mean GDP per capita (Burundi). For comparison, the GPD per capita of the United States in 2015 was $56,443.82. The bar plot below shows the 10 wealthiest nations (red bars) to the 10 poorest (blue) with GDP per capita shown on a log scale:

![GDP_Top_Countries]({{ site.url }}/images/GDP_Top_Countries.png)

Surprisingly, the feature with the largest positive correlation coefficient when compared to GDP per capita was percentage of population with internet access. This is not an unexpected correlation (we would expect wealthier nations to have more people with internet access), but it is surprising that the strongest correlation occurred with a non-economic feature. This suggests that economic features are not the only factors that can accurately model a nation's wealth. A scatterplot comparing GDP per capita to internet usage is shown below:

![GDP_vs_Internet]({{ site.url }}/images/GDP_vs_Internet.png)

On the other end, the feature with the highest-magnitude negative correlation coefficient was GVA by agriculture (GVA is a measure of economic productivity for a sector calculated by subtracting the cost of inputs/materials from the dollar amount of goods produced; reported in this dataset as % of GDP). This suggests that the wealthiest nations are primarily not agriculture-based with regard to their economic productivity. 

![GDP_vs_gva_ag]({{ site.url }}/images/GDP_vs_gva_ag.png)


### Multiple Linear Regression and Regularization

Ultimately to discern which features were most influential in capturing the discrepancies between the wealth of countries, I needed to model GDP per capita as a function of the other variables. Multiple linear regression was most suitable for this task, but I also implemented regularization techniques (namely ridge regression and LASSO) for comparison and feature selection. I started by splitting the data into train and test sets (using 2015 as the test set) and building a base linear regression model using *statsmodels* to examine key relationships between GDP and the predictors. 

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Train-test split with 2015 data as test
train = UN[(UN.Year == 2005) | (UN.Year == 2010)]
test = UN[UN.Year == 2015]

#Make design matrix with all features as predictors except log_gdp
y, X = patsy.dmatrices('log_gdp ~ exp_imp + gva_servc + labor_part + \
                       food_prod + life_exp + pop_gain + urban + log_pop_density + \
                       log_pop + log_gender_ratio + log_energy_prod + log_energy_supp +\
                       sqrt_internet + log_edu_prim + sqrt_gva_ag + sqrt_gva_indst + \
                       sqrt_unemp + log_tourism + log_health + sqrt_parliam_women + \
                       log_cpi + log_mort_child + log_mort_mother + log_num_child + \
                       log_pop_migrants + log_pop_unhcr + log_ghg_emissions', 
                       data = UN, return_type = "dataframe")

#Split predictor and response frames according to train-test grouping
X_train, X_test = X[X.index.isin(train.index)], X[X.index.isin(test.index)]
y_train, y_test = y[y.index.isin(train.index)], y[y.index.isin(test.index)]

#Fit base linear regression model with statsmodels
lm_base = sm.OLS(y_train, X_train)
fit_base = lm_base.fit()
fit_base.summary()

#Variance inflation factors for all predictors - multicollinearity definitely an issue
vif = [{X_train.columns[i]: variance_inflation_factor(np.array(X_train), i)} for i in range(X_train.shape[1])]
```

This model showed excellent fit (R2 = 0.954), but it was clear from the variance inflation factors that multicollinearity was an issue. Multicollinearity has the potential effect of inflating the standard errors of the coefficients and masking the true relationship between a predictor and the response. For example, if two predictors are strongly correlated, one might appear to have a significant effect on the response while the other is assigned a coefficient of near-zero. This presented a major challenge for interpreting linear regression models built with this dataset, as many of the features were inherently related and thus it was difficult to discern which features affected GDP and with what effect. I often found removing one feature from the model caused the coefficients of one or two others to change sign entirely! This is likely due to confounding or 'lurking' variables (an effect known as Simpson's Paradox): a feature may appear to have a positive relationship with the response when investigated in isolation, but the relationship may become negative when other variables are taken into account. An example is shown in the scatterplot below. While fertility rate (i.e. family size) appears to be negatively associated with GDP overall, it was consistently assigned a significant positive coefficient in linear regression models that included infant mortality rate, since the relationship becomes positive when the confounding variable is taken into account.

![GDP_vs_fertility_mortality]({{ site.url }}/images/GDP_vs_fertility_mortality.png) 

Nevertheless, the model was highly predictive of GDP per capita and appeared to adequately meet all of the regression assumptions. Residual diagnostic plots for the base model are shown:

![residual_plots]({{ site.url }}/images/residual_plots.png)

To manage multicollinearity/confounding variable issues, I implemented ridge and LASSO regularization as a means of feature selection and/or identification of the most important predictors. Both apply a penalty term to the linear regression cost function in order to shrink and/or remove coefficients that do not add substantial value to the model. In a way, both can thus dampen the effects of multicollinearity by identifying redundant features and giving them less weight in the model (LASSO commonly removes them entirely). Regularization generally requires standardizing the features first. To build linear regression, ridge, and LASSO models from different sets of features and compare their outputs efficiently, I coded the process as a series of functions:

```python
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def lr_lasso_ridge_Train_CV(predictors, response, k = 10, random_state = None):
    """
    Takes a dataframe of predictors and a dataframe of responses (training data), 
    finds optimal ridge and LASSO alpha values via cross-validation, conducts k-fold 
    cross-validation with k folds, fits linear regression, ridge, and LASSO models to
    the data, and returns a dataframe with the R^2, RMSE, and MAE values for both the
    training and validation sets of each fold for all three models. 
    
    Parameters
    -----------
    predictors: dataframe 
        predictor features as columns and training data as rows
    response: 1-column dataframe
        training data response values as rows
    k: int 
        number of folds to use for k-fold cross-validation (default is 10)
    random_state: int
        random state value to use for k-fold cross-validation (default is None) 
    
    Returns
    --------
    Dataframe with 18 rows and (k + 1) columns. Rows correspond to training and 
    validation set R^2 scores, RMSE values, and MAE values for linear regression (LR),
    LASSO (L1), and ridge (L2) models. Columns correspond to the kth fold, with the
    final column being the mean value of each row. 
    """
    
    #Find best alpha for Ridge and Lasso using k-fold CV on training set
    alphavec = 10**np.linspace(-2, 2, 200)
    lasso_best_alpha_model = LassoCV(alphas = alphavec, cv = k, random_state = random_state)
    lasso_best_alpha = lasso_best_alpha_model.fit(predictors, response.iloc[:, 0]).alpha_

    ridge_best_alpha_model = RidgeCV(alphas = alphavec, cv = k)
    ridge_best_alpha = ridge_best_alpha_model.fit(predictors, response.iloc[:, 0]).alpha_

    #Setup k-fold CV method
    kf = KFold(n_splits = k, shuffle = True, random_state = random_state)
    
    #Initialize empty vectors for metrics
    lr_r2_tr = []
    lr_r2_te = []
    lr_rmse_tr = []
    lr_rmse_te = []
    lr_mae_tr = []
    lr_mae_te = []
    
    l1_r2_tr = []
    l1_r2_te = []
    l1_rmse_tr = []
    l1_rmse_te = []
    l1_mae_tr = []
    l1_mae_te = []
    
    l2_r2_tr = []
    l2_r2_te = []
    l2_rmse_tr = []
    l2_rmse_te = []
    l2_mae_tr = []
    l2_mae_te = []
    
    for tr, val in kf.split(predictors, response):

        #Split into train and validation sets for each fold
        X_tr, X_val = predictors.iloc[tr], predictors.iloc[val]
        y_tr, y_val = response.iloc[tr], response.iloc[val]

        #Scale using the train set and apply scaling to train and val sets
        scale = StandardScaler().fit(X_tr)
        X_tr = scale.transform(X_tr)
        X_val = scale.transform(X_val)

        #Fit models, using best alphas for ridge and lasso
        lr = LinearRegression().fit(X_tr, y_tr)
        l1 = Lasso(alpha = lasso_best_alpha).fit(X_tr, y_tr)
        l2 = Ridge(alpha = ridge_best_alpha).fit(X_tr, y_tr)

        #Calculate R^2, RMSE, and MAE for each model/fold for train and val splits
        lr_r2_tr.append(r2_score(y_tr, lr.predict(X_tr)))
        lr_r2_te.append(r2_score(y_val, lr.predict(X_val)))
        lr_rmse_tr.append(np.sqrt(mean_squared_error(y_tr, lr.predict(X_tr))))
        lr_rmse_te.append(np.sqrt(mean_squared_error(y_val, lr.predict(X_val))))
        lr_mae_tr.append(mean_absolute_error(y_tr, lr.predict(X_tr)))
        lr_mae_te.append(mean_absolute_error(y_val, lr.predict(X_val)))
        
        l1_r2_tr.append(r2_score(y_tr, l1.predict(X_tr)))
        l1_r2_te.append(r2_score(y_val, l1.predict(X_val)))
        l1_rmse_tr.append(np.sqrt(mean_squared_error(y_tr, l1.predict(X_tr))))
        l1_rmse_te.append(np.sqrt(mean_squared_error(y_val, l1.predict(X_val))))
        l1_mae_tr.append(mean_absolute_error(y_tr, l1.predict(X_tr)))
        l1_mae_te.append(mean_absolute_error(y_val, l1.predict(X_val)))
        
        l2_r2_tr.append(r2_score(y_tr, l2.predict(X_tr)))
        l2_r2_te.append(r2_score(y_val, l2.predict(X_val)))
        l2_rmse_tr.append(np.sqrt(mean_squared_error(y_tr, l2.predict(X_tr))))
        l2_rmse_te.append(np.sqrt(mean_squared_error(y_val, l2.predict(X_val))))
        l2_mae_tr.append(mean_absolute_error(y_tr, l2.predict(X_tr)))
        l2_mae_te.append(mean_absolute_error(y_val, l2.predict(X_val)))
        
    #Store metrics in dataframe output
    metrics = pd.DataFrame([lr_r2_tr, lr_r2_te, lr_rmse_tr, lr_rmse_te, lr_mae_tr, lr_mae_te, 
                            l1_r2_tr, l1_r2_te, l1_rmse_tr, l1_rmse_te, l1_mae_tr, l1_mae_te, 
                            l2_r2_tr, l2_r2_te, l2_rmse_tr, l2_rmse_te, l2_mae_tr, l2_mae_te],
                           index = ['lr_r2_tr', 'lr_r2_te', 'lr_rmse_tr', 'lr_rmse_te', 
                                    'lr_mae_tr', 'lr_mae_te', 'l1_r2_tr', 'l1_r2_te', 
                                    'l1_rmse_tr', 'l1_rmse_te', 'l1_mae_tr', 'l1_mae_te', 
                                    'l2_r2_tr', 'l2_r2_te', 'l2_rmse_tr', 'l2_rmse_te', 
                                    'l2_mae_tr', 'l2_mae_te'])
    
    #Calculate means across folds
    metrics['means'] = metrics.apply(np.mean, axis = 1)
    return metrics


def lr_lasso_ridge_Train_Test(pred_train, pred_test, resp_train, resp_test, k = 10, random_state = None):
    """
    Takes 4 dataframes corresponding to the predictor training set, predictor test set, 
    response training set, and response test set, finds optimal ridge and LASSO alpha values 
    via cross-validation on the training set, fits linear regression, ridge, and LASSO models to
    the training data, and returns printed output reporting the R^2, RMSE, and MAE for both the 
    training and test sets for all three models. 
    
    Parameters
    -----------
    pred_train: dataframe 
        predictor features as columns and training data as rows
    pred_test: dataframe 
        predictor features as columns and test data as rows
    resp_train: 1-column dataframe
        training data response values as rows
    resp_test: 1-column dataframe
        test data response values as rows
    k: int 
        number of folds to use for ridge and LASSO cross-validation (default is 10)
    random_state: int
        random state value to use for ridge and LASSO cross-validation (default is None) 
    
    Returns
    --------
    18 lines of printed text output reporting the R^2 score, RMSE value, and MAE value
    for the training and test data of the linear regression (lin reg), LASSO (L1), and 
    ridge (L2) models. 
    """
    
    #Find best alpha for Ridge and Lasso using k-fold CV on training set
    alphavec = 10**np.linspace(-2, 2, 200)
    lasso_best_alpha_model = LassoCV(alphas = alphavec, cv = k, random_state = random_state)
    lasso_best_alpha = lasso_best_alpha_model.fit(pred_train, resp_train.iloc[:, 0]).alpha_

    ridge_best_alpha_model = RidgeCV(alphas = alphavec, cv = k)
    ridge_best_alpha = ridge_best_alpha_model.fit(pred_train, resp_train.iloc[:, 0]).alpha_
    
    #Scale using training set and apply scaling to train and test sets
    scale = StandardScaler().fit(pred_train)
    pred_train_scale = scale.transform(pred_train)
    pred_test_scale = scale.transform(pred_test)

    #Fit models
    lr_model = LinearRegression().fit(pred_train_scale, resp_train)
    l1_model = Lasso(alpha = lasso_best_alpha).fit(pred_train_scale, resp_train)
    l2_model = Ridge(alpha = ridge_best_alpha).fit(pred_train_scale, resp_train)
    
    #Print R^2, RMSE, and MAE values for train and test sets
    print('lin reg train R2: ', r2_score(resp_train, lr_model.predict(pred_train_scale)))
    print('lin reg test R2: ', r2_score(resp_test, lr_model.predict(pred_test_scale)), '\n')
    print('lin reg train RMSE: ', np.sqrt(mean_squared_error(resp_train, lr_model.predict(pred_train_scale))))
    print('lin reg test RMSE: ', np.sqrt(mean_squared_error(resp_test, lr_model.predict(pred_test_scale))), '\n')
    print('lin reg train MAE: ', mean_absolute_error(resp_train, lr_model.predict(pred_train_scale)))
    print('lin reg test MAE: ', mean_absolute_error(resp_test, lr_model.predict(pred_test_scale)), '\n')
    print('lasso train R2: ', r2_score(resp_train, l1_model.predict(pred_train_scale)))
    print('lasso test R2: ', r2_score(resp_test, l1_model.predict(pred_test_scale)), '\n')
    print('lasso train RMSE: ', np.sqrt(mean_squared_error(resp_train, l1_model.predict(pred_train_scale))))
    print('lasso test RMSE: ', np.sqrt(mean_squared_error(resp_test, l1_model.predict(pred_test_scale))), '\n')
    print('lasso train MAE: ', mean_absolute_error(resp_train, l1_model.predict(pred_train_scale)))
    print('lasso test MAE: ', mean_absolute_error(resp_test, l1_model.predict(pred_test_scale)), '\n')
    print('ridge train R2: ', r2_score(resp_train, l2_model.predict(pred_train_scale)))
    print('ridge test R2: ', r2_score(resp_test, l2_model.predict(pred_test_scale)), '\n')
    print('ridge train RMSE: ', np.sqrt(mean_squared_error(resp_train, l2_model.predict(pred_train_scale))))
    print('ridge test RMSE: ', np.sqrt(mean_squared_error(resp_test, l2_model.predict(pred_test_scale))), '\n')
    print('ridge train MAE: ', mean_absolute_error(resp_train, l2_model.predict(pred_train_scale)))
    print('ridge test MAE: ', mean_absolute_error(resp_test, l2_model.predict(pred_test_scale)))
```

The first takes a training set, performs k-fold cross-validation with a range of alpha values to determine the optimal alpha for ridge and LASSO, then performs k-fold cross-validation again and iterates over every train-validation split. For each fold, it scales the training split, fits linear regression, LASSO, and ridge models (with optimal alphas) to the training split, calculates model evaluation metrics for the training and validation splits, and appends these to a dataframe. The second conducts a similar procedure but only uses supplied training and test sets rather than k-fold cross-validation splits, and simply prints the evaluation results rather than compiling them in a dataframe. 

I also defined functions for calculating AIC, BIC, and adjusted R2 statistics for use with linear regression models, since these are not included in the standard *sklearn* model output:

```python
def aic(y_obs, y_pred, p):
    """
    Takes two 1-D arrays or 1-column dataframes representing the 
    observed and predicted response values of a linear regression 
    training dataset, as well as the number of model parameters (p), 
    and returns the AIC statistic calculated using the following 
    formula:
    
    (n)ln(RSS) - (n)ln(n) + 2p
    
    Parameters
    ----------
    y_obs: 1-column dataframe or 1-D array
        observed ('true') response values of training dataset
    y_pred: 1-column dataframe or 1-D array
        response values of training dataset predicted using linear 
        regression
    p: int
        number of parameters of linear regression model (including 
        intercept)
    
    Returns
    -------
    AIC statistic (float) of linear regression model.
    """
    n = len(y_obs)
    SSE = np.sum((np.array(y_obs) - np.array(y_pred))**2)
    return (n * np.log(SSE)) - (n * np.log(n)) + (2 * p)


def bic(y_obs, y_pred, p):
    """
    Takes two 1-D arrays or 1-column dataframes representing the 
    observed and predicted response values of a linear regression 
    training dataset, as well as the number of model parameters (p), 
    and returns the BIC statistic calculated using the following 
    formula:
    
    (n)ln(RSS) - (n)ln(n) + (p)ln(n)
    
    Parameters
    ----------
    y_obs: 1-column dataframe or 1-D array
        observed ('true') response values of training dataset
    y_pred: 1-column dataframe or 1-D array
        response values of training dataset predicted using linear 
        regression
    p: int
        number of parameters of linear regression model (including 
        intercept)
    
    Returns
    -------
    BIC statistic (float) of linear regression model.
    """
    n = len(y_obs)
    SSE = np.sum((np.array(y_obs) - np.array(y_pred))**2)
    return (n * np.log(SSE)) - (n * np.log(n)) + (p * np.log(n))


def adj_r2(y_obs, y_pred, p):
    """
    Takes two 1-D arrays or 1-column dataframes representing the 
    observed and predicted response values of a linear regression 
    training dataset, as well as the number of model parameters (p), 
    and returns the Adjusted R^2 score.
    
    Parameters
    ----------
    y_obs: 1-column dataframe or 1-D array
        observed ('true') response values of training dataset
    y_pred: 1-column dataframe or 1-D array
        response values of training dataset predicted using linear 
        regression
    p: int
        number of parameters of linear regression model (including 
        intercept)
    
    Returns
    -------
    Adjusted R^2 score (float) of linear regression model.
    """
    n = len(y_obs)
    SSE = np.sum((np.array(y_obs) - np.array(y_pred))**2)
    SST = np.sum((np.array(y_obs) - float(np.mean(y_obs)))**2)
    return 1 - ((SSE / (n - p - 1)) / (SST / (n - 1)))
```

Using these functions in conjunction with a variety of training datasets hosting different combinations of features, I found that regularization offered little-to-no improvement over general linear regression in test set performance, but it did shed light on important and redundant features. Ultimately I reduced the feature space to include only the following predictors:

* Labor force participation rate
* Food Production Index
* Life expectancy
* Population density
* Energy supplied per capita
* Percentage of population with internet access
* GVA by agriculture
* Unemployment rate
* Tourism expenditure
* Fertility rate
* GHG emissions per capita

Others were deemed redundant due to multicollinearity or insignificant in modelling the response (high p-value from t-test of coefficient). This does not necessarily imply that they do not have a relationship with the response; it simply means they were not needed in a model that already included these eleven predictors. 


### Final Remarks

From the regression models, it seems wealthy nations could be described by the following general characteristics:

1. Better access to healthcare and technology resources (e.g. internet, hospitals, etc.)
2. Larger energy consumption per capita (strongly associated with higher GHG emissions)
3. Service-driven economies rather than agriculture-driven
4. Large migrant and urban populations
5. Large tourism expenditures
6. Low infant and maternal mortality rates (tied to healthcare access); high life expectancy
7. Large family sizes (dependent on other parameters)
8. Low population densities
9. Low unemployment and labor force participation rates

Factors that were largely not predictive of GDP:

1. Export - import differential
2. Year-to-year population gain
3. Total population
4. Gender ratio
5. Energy produced per capita
6. Percentage of GDP spent on healthcare
7. Percentage of parliament comprised of women
8. Number of students enrolled in primary education 

Most of these were fairly predictable, but I found a few of them somewhat surprising- namely the negative relationship with labor force participation rate and the general lack of association with education and energy production. These would need to be explored further to establish whether there may be additional lurking variables or collinearity issues present. 

All in all, I believe this analysis shows that while economic indicators are the primary drivers behind wealth on a global scale, they are not the only metrics for evaluating a nation's economic health. There are seemingly valuable demographic and social metrics that contribute to the complexities of the system.     
