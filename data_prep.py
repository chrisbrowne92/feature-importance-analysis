import numpy as np
import pandas as pd
import myfuncs as my
from sklearn.impute import SimpleImputer

# import survey data, monthly consumption & customer list files
survey = 'baseline'
update_from_other_survey = False
survey_update = 'midline'
routeDir = ''
filesToImport = ["appliance_customer_list.csv",
                 "aggregated_consumption_daily.csv",
                 'appliance_' + survey + '.csv',
                 'appliance_' + survey_update + '.csv',
                 'appliance_headings.csv']

completeness_threshold = 0.9

# for completeness_threshold in completeness_thresholds:
[customers, consumption, data, data_update, questions] = my.get_data_from_csvs(routeDir,
                                                                               filesToImport)  # import data from CSVs
questions = questions.set_index('Master_Heading').drop('Index', axis=1)  # index the questions by the master heading
questions.loc[:, 'Model'] = questions.loc[:, 'Model'].fillna(False)  # fill blanks as false in model column

## DATA PREPARATION
# keep only data and columns to be used in the modelling
categorical_groups = list(questions[questions['Heading_name_' + survey].notnull() &
                                    (questions['Data_type'] == 'ct_bool')]
                          .loc[:, 'Heading_name_' + survey])  # save list of categorical q's for later
questions = questions[questions['Heading_name_' + survey].notnull() &
                      questions['Model']]  # keep only questions appearing in this survey, and flagged to go into models
data = data.loc[data['Meter_number'].notnull(), :]  # remove lines without meter number
exchange_rates = data['Country'].map(my.EXCHANGE_RATE_TO_USD)  # map exchange rates on country for later
exchange_rates.index = data['Meter_number']
data = data.set_index('Meter_number').loc[:, questions['Heading_name_' + survey]]  # keep only columns for models
(N, M) = data.shape  # get number of samples and columns

# sum energy expenditure into one variable
energy_questions = questions.index.values[questions.index.str.endswith('_Monthly_Expenses')]
new_question = 'Monthly_Energy_Expenses'
data[new_question] = data.loc[:, energy_questions].sum(axis=1, min_count=1)
data, questions = my.remove_questions(data, questions, energy_questions)
questions = my.add_questions(questions, new_questions=[new_question], data_type='currency')

# determine questions completeness (percentage of survey submissions that answered a question)
completeness = pd.DataFrame()
completeness[survey] = data.loc[:, questions[questions['Model']]['Heading_name_' + survey]].count() / N

# Replace missing baseline data with midline data
if update_from_other_survey:
    column_mapping = dict(zip(questions['Heading_name_' + survey_update], questions['Heading_name_' + survey]))
    data_update = data_update.rename(column_mapping, axis='columns')
    (N_cust, M_cust) = customers.shape  # get size of customer list
    meters = customers[survey + '_response_under_meter_num']  # meters in data
    meters_updated = customers[survey_update + '_response_under_meter_num']  # equivalent meters in update survey
    meter_mapping = dict([(meters_updated[i], meters[i])
                          for i in range(N_cust)
                          if meters[i] != 'NONE'
                          and meters_updated[i] != 'NONE'])  # create mapping of meters from update to original survey
    data_update = data_update.set_index('Meter_number')
    data = data.set_index('Meter_number')
    data_update = data_update.rename(meter_mapping, axis='index')
    data.update(data_update, overwrite=False)
    completeness['updated with ' + survey_update] = data.loc[:, questions['Heading_name_' + survey]].count() / N

# remove questions that are below a completeness threshold
# completeness_threshold = 0.95
questions_to_remove = completeness[survey][completeness[survey] <= completeness_threshold].index.values
data, questions = my.remove_questions(data, questions, questions_to_remove)

# parse time variables, and make them cyclical (i.e. 23 is as close to 00 as it is to 22)
time_questions = questions[questions['Data_type'] == 'time']['Heading_name_' + survey]
times = data.loc[:, time_questions]
for col in times.columns.values:
    times[col] = pd.to_numeric(times.loc[:, col].str.slice(0, 2), errors='coerce') + \
                 pd.to_numeric(times.loc[:, col].str.slice(3, 5), errors='coerce') / 60
    data[col + '_sin'] = np.sin(times[col] * (2. * np.pi / 24))
    data[col + '_cos'] = np.cos(times[col] * (2. * np.pi / 24))
    # update questions file for new columns
    data, questions = my.remove_questions(data, questions, [col])
    questions = my.add_questions(questions, new_questions=[col + '_sin', col + '_cos'], data_type='cycle')

# one-hot encoding of categorical variables
one_hot_questions = questions[questions['Data_type'] == 'catgry']['Heading_name_' + survey]
data = pd.get_dummies(data, columns=one_hot_questions)  # one-hot encode columns
questions = questions.drop(one_hot_questions, axis=0)  # drop old questions
new_questions = pd.Series(
    data.columns.values[data.columns.str.contains('|'.join(one_hot_questions))])  # determine new questions
new_questions = new_questions[~new_questions.isin(questions.index.values)]  # not including existing questions
questions = my.add_questions(questions, new_questions=new_questions, data_type='binary')  # add to questions list

# group occupations together
occupation_groups = set(my.OCCUPATION_GROUPS.values())  # get list of unique occupation groups
# get questions related to occupation
occupation_questions = questions.index.values[questions.index.str.startswith('Occupation__')]
for group in set(my.OCCUPATION_GROUPS.values()):
    data['Occupation__' + group] = my.or_of_columns(data, [occp for occp in occupation_questions
                                                           if my.OCCUPATION_GROUPS[occp[12:]] == group])
data = data.drop(occupation_questions, axis=1)
questions = questions.drop(occupation_questions, axis=0)
questions = my.add_questions(questions, new_questions=['Occupation__' + a for a in occupation_groups], data_type='bool')

# normalise currencies to USD
# exchange_rates = data['Country'].map(my.EXCHANGE_RATE_TO_USD)   # done above
currency_questions = questions[questions['Data_type'] == 'currency']['Heading_name_' + survey]
data[currency_questions] = data[currency_questions].multiply(exchange_rates, axis=0)

# change boolean to 1/0
mask = (questions['Data_type'] == 'bool') & (questions['Heading_name_' + survey].notnull())
questions.loc[mask, 'Data_type'] = 'binary'  # set data type to binary
bool_questions = questions[mask]['Heading_name_' + survey]  # get questions that are bool data_type
data.loc[:, bool_questions] = data[bool_questions].astype(float)  # set values to 1/0

# group categorical answers with <5% responses
# get list of questions which are, or become, one-hot encoded
# a group is the questions for one particular question
groups = list(one_hot_questions) + categorical_groups
T = N * 0.05  # 5% sample threshold
for group in groups:
    group_questions = questions.index.values[questions.index.str.startswith(group + '_') &
                                             (questions['Data_type'] == 'binary')]
    group_responses = data[group_questions].sum(axis=0, skipna=True)  # count the ones (i.e. responses in that column)
    low_freq_questions = group_responses.index.values[
        group_responses < T]  # take questions which are below 5% responses
    if group not in one_hot_questions:
        a = 1
    if len(low_freq_questions) > 1:  # if there is more than one question that contains <5% responses, merge
        temp = data[low_freq_questions].sum(axis=1)
        data[group + '__other_low_freq'] = data[low_freq_questions].sum(axis=1)  # add together the ones
        data = data.drop(low_freq_questions, axis=1)
        questions = questions.drop(low_freq_questions, axis=0)
        questions = my.add_questions(questions, new_questions=[group + '__other_low_freq'],
                                     data_type='binary')

questions = questions.sort_index()

# impute mean in place of nan values
imp = SimpleImputer(strategy='mean')
cols = data.columns
idx = data.index
data = pd.DataFrame(imp.fit_transform(data))
data.index = idx
data.columns = cols

questions.to_csv(routeDir + 'questions_prepped.csv')

## build dependent variable: average daily consumption Oct-Dec 2017
# get consumption for customers who responded to survey
consumption = my.get_consumption_for_customers(consumption, customers, [survey])
months = ['17-Oct', '17-Nov', '17-Dec']
cols = consumption.columns.str.contains('|'.join(months))  # find columns for desired months
consumption = consumption.loc[:, cols]  # ignore all other data
consumption_average = consumption.mean(axis=1)  # find mean consumption for each meter
# data = data.set_index('Meter_number')  # set meter number as the index before merging
data['Consumption'] = consumption_average  # add consumption as a new column

data.to_csv(routeDir + 'data_prepped.csv')
