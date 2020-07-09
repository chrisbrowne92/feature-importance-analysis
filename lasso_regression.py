import numpy as np
import pandas as pd
import myfuncs as my
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

# import survey data, monthly consumption & customer list files
survey = 'baseline'
update_from_other_survey = False
survey_update = 'midline'
routeDir = ''
filesToImport = ['questions_prepped.csv',
                 'data_prepped.csv']

# for completeness_threshold in completeness_thresholds:
[questions, data] = my.get_data_from_csvs(routeDir, filesToImport)  # import data from CSVs
data = data.set_index('Meter_number')

plt_dir = ''

# remove customers with NaN consumption
data = data[data['Consumption'].notnull()]

# define lasso mode and data
lasso = Lasso()
labels = np.array(data['Consumption'])
features = np.array(data.drop('Consumption', axis=1))

# split data in to training/test
train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                            test_size=0.25, random_state=42)
# fit lasso to training data
lasso.fit(train_features, train_labels)

# test on training data
predictions = lasso.predict(test_features)

# Calculate the absolute errors
errors = abs(predictions - test_labels)
non_zero = (test_labels != 0)
errors_percent = 100 * errors[non_zero] / test_labels[non_zero]

# put results together in dataframe
results = pd.DataFrame(dict(zip(['actual', 'prediction', 'error'],
                                [test_labels, predictions, errors])))
results.to_clipboard()

# extract LASSO coefficients
coefs = pd.Series(lasso.coef_, index=data.columns.values[:-1])
non_zero = (abs(coefs) != 0)
coefs = coefs[non_zero]

# Print out the mean absolute error (mae)
print('Num coefs used:', np.sum(lasso.coef_ != 0))
print('Mean Absolute Error:', round(np.mean(errors), 2))
print('Mean Squared Error:', round(np.mean(np.square(errors)), 2))
print('Median Absolute Error:', round(np.median(errors), 2))
print('Median Absolute Percentage Error:', round(np.median(errors_percent), 2), '% (does not include 0 consumption)')
print('Aggregate error: ', round(100 * ((np.sum(test_labels) - np.sum(predictions)) / np.sum(test_labels)), 2), '% (does not include 0 consumption)')
