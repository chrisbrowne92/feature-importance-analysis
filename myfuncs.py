import numpy as np
from numpy.linalg import lstsq
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot
from scipy import stats

plot_dir = ''

######################### Constants #########################

# exchange rates
EXCHANGE_RATE_TO_USD = {'kenya': 1 / 100,
                        'tanzania': 1 / 2222}

MONTHS_TO_IDX = {'Jan': 1,
                 'Feb': 2,
                 'Mar': 3,
                 'Apr': 4,
                 'May': 5,
                 'Jun': 6,
                 'Jul': 7,
                 'Aug': 8,
                 'Sep': 9,
                 'Oct': 10,
                 'Nov': 11,
                 'Dec': 12}

MONTHS_TO_NUM_DAYS = {'Jan': 31,
                      'Feb': 28,
                      'Mar': 31,
                      'Apr': 30,
                      'May': 31,
                      'Jun': 30,
                      'Jul': 31,
                      'Aug': 31,
                      'Sep': 30,
                      'Oct': 31,
                      'Nov': 30,
                      'Dec': 31}

MONTHS_NUM_TO_STRING = {1: 'Jan',
                        2: 'Feb',
                        3: 'Mar',
                        4: 'Apr',
                        5: 'May',
                        6: 'Jun',
                        7: 'Jul',
                        8: 'Aug',
                        9: 'Sep',
                        10: 'Oct',
                        11: 'Nov',
                        12: 'Dec'}

OCCUPATION_GROUPS = {'comm_farmer': 'Agriculture',
                     'subs_farmer': 'Agriculture',
                     'farm_labor': 'Day-labour',
                     'housework': 'Day-labour',
                     'maid': 'Day-labour',
                     'non_farm_labor': 'Day-labour',
                     'carpenter': 'Business',
                     'hairdresser': 'Business',
                     'mechanic': 'Business',
                     'miller': 'Business',
                     'restaurateur': 'Business',
                     'shop': 'Business',
                     'tailor': 'Business',
                     'technician': 'Business',
                     'driver': 'Services',
                     'healthcare': 'Services',
                     'military': 'Services',
                     'police': 'Services',
                     'public_official': 'Services',
                     'teacher': 'Services',
                     'other': 'Other',
                     'none': 'None'}


######################### Data Manilupation #########################

def get_data_from_csvs(route_dir, files_to_import):
    # from list in files_to_import, import data files into DataFrames and return as a list
    files_to_import = [route_dir + item for item in files_to_import]
    return [pd.read_csv(fl, delimiter=",", skiprows=0, low_memory=False) for fl in files_to_import]


def get_consumption_for_customers(consumption, customers, surveys, merge_changed_meters=True):
    # returns the consumption for the input customers
    # surveys - list of survey names that you are interested in getting data from
    # merge_old_and_new - used to merge the consumption for meter numbers that has changed
    #                     True: merge,  False: don't merge
    # generate a list of unique meter numbers for the all submissions in all surveys
    meters_in_surveys = []
    for survey in surveys:
        meters_in_surveys = meters_in_surveys + [customers[survey + '_response_under_meter_num'][
                                                     customers[survey + '_response_under_meter_num'] != 'NONE']]
    unique_meters_in_surveys = set(pd.concat(meters_in_surveys))
    # keep only consumption records for meters with survey responses
    consumption = consumption[consumption['MeterNumber'].isin(unique_meters_in_surveys)]
    if merge_changed_meters:  # merge consumption for changed meter numbers
        # merge (sum) consumption for old/new meter numbers
        consumption = merge_consumption(consumption, customers, merge_old_and_new=True)
    return consumption


def consumption_gradients(consumption, smoothing):
    if smoothing:
        # smooth consumption
        pass  # do nothing, not written yet
    # fit a line with least squares and determine gradient
    pass


def merge_rows_by_summing(df, by_columns):
    # merges rows in a DataFrame df which have values in the column by_column that are the same.
    # where there are repeated values in by_column, the data in all rows is summed
    # i.e. if there's data in a the same column of two rows to be merged, it sums the data
    # df - DataFrame with some repeated indexes
    # merge_by_column - string, name of the column by which to merge
    # sum_columns - sum data in these columns during merge. all other columns will be put into index and not merged
    return df.groupby(by_column, sort=True).agg(pd.Series.sum, min_count=1, numeric_only=True)


def merge_consumption(consumption, customers, merge_old_and_new=False, merge_multiple_meters=False):
    # merges consumption for meter numbers that have changed, by summing them together
    # inputs:
    # consumption - DataFrame with columns as time periods, indexed by meter number, and values as consumption
    # customers - DataFrame containing information for customers in 'consumption'
    # customers_changed - DataFrame of meter numbers that have changed, such that
    #      customers_changed['Current_Meter_Number'][i] = new meter number
    #      customers_changed['Old_Meter_Number'][i] = old meter number
    # multiple_meters - dict containing meter number for each customer where the key is the root connection
    #      e.g. multiple_meters[<root meter number>] = list of other meter numbers for that customer
    if merge_old_and_new:
        customers_changed = customers[['Current_Meter_Number', 'Old_Meter_Number']][
            customers['Old_Meter_Number'].notnull()]
        new = customers_changed['Current_Meter_Number']
        old = customers_changed['Old_Meter_Number']
        if consumption.index.name == 'MeterNumber':
            consumption = consumption.reset_index()  # put meter number back into columns
        mask = consumption['MeterNumber'].isin(old)  # create mask to select only rows for old meter numbers
        consumption.loc[mask, 'MeterNumber'] = new  # rename old meter number to new meter number before merging
        consumption = merge_rows_by_summing(consumption, 'MeterNumber')  # merge and sum meter numbers
    if merge_multiple_meters:  # merge all meter numbers with the same root
        headings = ['Current_Meter_Number', 'Root_Meter_Num']  # headings to keep
        customers_multiple_meters_IDs = customers.duplicated(subset='ID', keep=False)
        # select only customers that have multiple meters, but aren't the root connection
        customers_multiple_meters = customers[headings][customers.duplicated(subset='ID', keep=False) &
                                                        ~customers['Root_Connection']]
        current = customers_multiple_meters['Current_Meter_Number']  # current meter number
        root = customers_multiple_meters['Root_Meter_Num']  # root meter numbers
        if consumption.index.name == 'MeterNumber':
            consumption = consumption.reset_index()  # put meter number back into columns
        mask = consumption['MeterNumber'].isin(current)  # mask to select only rows which are not root connections
        consumption.loc[mask, 'MeterNumber'] = root  # rename current meter number to root meter number before merging
        consumption = merge_rows_by_summing(consumption, 'MeterNumber')  # merge and sum meter numbers
    return consumption


def split_customers_by_appliance_ownership(customer_list):
    # splits customer list into those who received appliances, and those that didn't
    # returns dictionary with
    # keys: group name
    # values: the root connection of each customer, i.e. multiple connections are ignored
    control = customer_list[~customer_list.This_customer_received_appliance & customer_list.Root_Connection]
    treatment = customer_list[customer_list.This_customer_received_appliance & customer_list.Root_Connection]
    groups = {'control': control, 'treatment': treatment}
    return groups


def split_data_into_customer_groups(surveys_in, customer_groups):
    # splits survey data into the customer groups defined by customer_groups
    # returns a dict containing the survey data for each customer group,
    # referenced as surveys_out[survey][group], e.g. surveys_out['baseline']['control']
    #
    # surveys_in: dictionary with keys: survey names, values: DataFrame of survey data.
    # customer_groups: dictionary with keys: group names, values: DataFrame of customer data (meter num, responses etc)
    surveys_out = {}  # dictionary containing the split data for all surveys from surveys_in,
    # each survey (themselves a dictionary) is split into customer groups
    for survey_name, survey_data in surveys_in.items():  # loop through surveys
        if survey_data is not None:  # check if there's data
            survey_data_in_groups = {}  # dict to contain the data from each customer group for a particular survey
            for group_name, group_customers in customer_groups.items():
                # filter the survey_data to only include meter numbers from the group_customers
                data_for_group = survey_data[
                    survey_data.Meter_number.isin(group_customers[survey_name + '_response_under_meter_num'])]
                # add dictionary entry for the data under the group_name
                survey_data_in_groups[group_name] = data_for_group
            surveys_out[survey_name] = survey_data_in_groups  # store in output dictionary
            survey_data_in_groups = survey_data_in_groups.copy()  # copy so existing dict isn't modified next iteration
        else:
            surveys_out[survey_name] = None  # set survey to none, if there's no data
    return surveys_out


def remove_questions(data, headings, questions_to_remove):
    # removes questions from data (in columns) and headings (in rows) variables
    data = data.drop(questions_to_remove, axis=1)
    headings = headings.drop(questions_to_remove, axis=0)
    return data, headings


def corr_with_significance(df):
    mat = df.values.T
    K = len(df.columns)
    correl = np.empty((K, K), dtype=float)
    p_vals = np.empty((K, K), dtype=float)

    for i, ac in enumerate(mat):
        for j, bc in enumerate(mat):
            if i > j:
                continue
            else:
                corr = stats.pearsonr(ac, bc)

            correl[i, j] = corr[0]
            correl[j, i] = corr[0]
            p_vals[i, j] = corr[1]
            p_vals[j, i] = corr[1]

    pvalues = pd.DataFrame(p_vals)
    pvalues.columns = df.columns
    pvalues.index = df.columns
    corrr = pd.DataFrame(correl)
    corrr.columns = df.columns
    corrr.index = df.columns
    return corrr, pvalues


######################### General #########################

def replace_value_in_column(series, value_to_change, change_to):
    series[series == value_to_change] = change_to
    return series


def date_cols_to_num_days(df, col_names, start_date, format="%d/%m/%Y"):
    for col in col_names:
        df[col] = date_string_to_num_days(df[col], start_date, format)
    return df


def date_string_to_num_days(date_strings, start_date, format="%d/%m/%Y"):
    out = (pd.to_datetime(date_strings, format=format) - start_date).dt.days
    return out


def is_in_list(lst, val):
    return lst.count(val) > 0


def drop_cols(df, cols):
    for col in df.columns.values:
        if col in cols:
            df = df.drop(col, axis=1)
    return df


def keep_only_cols(df, cols):
    for col in df.columns.values:
        if col not in cols:
            df = df.drop(col, axis=1)
    return df


def complete_rows(df, columns=None):
    if columns is not None:
        df = keep_only_cols(df, columns)
    mask = ~pd.isnull(df).any(1)
    return mask


def or_of_columns(df, columns):
    # performs element-wise or of list of columns in data frame
    # returns Series of True/False/NaN
    N = len(df[columns[0]])  # get number of samples
    result = pd.Series(np.array([np.nan] * N))  # initialise array to nans to preserve unanswered questions
    mask = df[columns[0]].notnull()  # mask for questions that are answered
    result.index = mask.index
    falses = pd.Series(np.zeros(N).astype(bool))  # array of
    falses.index = mask.index
    result[mask] = falses[mask]  # set masked questions to false
    for col in columns:
        result = result | df[col]  # element-wise logical or of columns
    return result


def add_questions(headings, new_questions, data_type, model=True, psm=True):
    for heading in new_questions:
        new_row = {'Master_Heading': [heading],
                   'Data_type': [data_type],
                   'Model': [model],
                   'PSM': [psm],
                   'Heading_name_baseline': [heading],
                   'Heading_name_midline': [heading]}
        new_row = pd.DataFrame.from_dict(new_row)
        new_row = new_row.set_index('Master_Heading')
        headings = pd.concat([headings, new_row], sort=False)
    return headings


def between_vals(series, a, b, return_mask=False):
    mask = (series >= a) & (series < b)
    if return_mask:
        return mask
    else:
        return series[mask]


def round_to(num, divisor):
    return round(num / divisor) * divisor


def round_to_sig_fig(num, sig_fig):
    return round_to(num, np.power(10,np.floor(np.log10(num))-(sig_fig-1)))


######################### Plotting #########################

def distplot_multiple(allSeries, seriesLabels=None, bins=20, filters=None, kde=False, rug=False, hist=True,
                      norm_hist=True, xLabel=None, yLabel=None, plot_title=None, show_plot=False, save_plot=False,
                      save_filename=None, clear_plot=False):
    i = 0
    for srs in allSeries:
        if seriesLabels == None:
            srsLabels = None
        else:
            srsLabels = seriesLabels[i]
        if filters is not None:
            sns.distplot(srs[filters[i]], bins=bins, kde=kde, rug=rug, hist=hist, norm_hist=norm_hist, label=srsLabels)
        else:
            sns.distplot(srs, bins=bins, kde=kde, rug=rug, hist=hist, norm_hist=norm_hist, label=srsLabels)
        i += 1
    plt.legend()
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(plot_title)
    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig(plot_dir + save_filename, bbox_inches='tight')
    if clear_plot:
        plt.clf()


def barplot_multiple(dictOfSeries, dictOfErrors=None, plot_title=None, show_plot=False, save_plot=False,
                     save_filename=None, clearPlot=True):
    numBars = len(dictOfSeries.values())
    numCategories = len(dictOfSeries[next(iter(dictOfSeries))])
    barWidth = 1
    bar_pos_start = -((numBars / 2) - 0.5) * barWidth
    categorySpacing = barWidth / 2
    categoryWidth = barWidth * numBars + categorySpacing
    label_pos = np.arange(numCategories) * categoryWidth
    i = 0
    for key in dictOfSeries.keys():
        bar_pos = label_pos + bar_pos_start + i * barWidth
        plt.bar(bar_pos, dictOfSeries[key].values(), align='center', alpha=0.5, width=barWidth)
        if dictOfErrors is not None:
            plt.errorbar(bar_pos, dictOfSeries[key].values(), yerr=dictOfErrors[key].values())
        i += 1
    plt.xticks(label_pos, dictOfSeries[key].keys())
    plt.xticks(rotation=90)
    plt.ylabel('Quantity')
    plt.legend(dictOfSeries.keys(), loc=1)
    # plt.legend()
    plt.title(plot_title)
    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig('plots/' + save_filename, bbox_inches='tight')
    if clearPlot:
        plt.clf()


def barplot_single(series, dictOfErrors=None, plot_title=None, rotate_labels=False,
                   show_plot=False, save_plot=False, save_filename=None, clearPlot=True,
                   ylabel=None, xlabel=None, skip_ticks=1):
    num_categories = len(series)
    bar_width = 1
    category_spacing = bar_width / 2
    category_width = bar_width + category_spacing
    label_pos = np.arange(num_categories) * category_width
    plt.bar(label_pos, series, align='center', alpha=0.5, width=bar_width)
    if dictOfErrors is not None:
        plt.errorbar(label_pos, series, yerr=dictOfErrors.values())
    # plt.xticks(label_pos, series.index.values)
    plt.xticks(label_pos[0::skip_ticks], series.index.values[0::skip_ticks])
    if rotate_labels:
        plt.xticks(rotation=90)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_title)
    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig('plots/' + save_filename, bbox_inches='tight')
    if clearPlot:
        plt.clf()


def lineplot(x, y, dashed_line=False, plot_title=None, show_plt=False, save_plot=False, save_file_name=None,
             clear_plot=True):
    if dashed_line:
        plt.plot(x, y, ':')
    else:
        plt.plot(x, y)
    plt.title(plot_title)
    if show_plt:
        plt.show()
    if save_plot:
        plt.savefig('plots/' + save_file_name, bbox_inches='tight')
    if clear_plot:
        plt.clf()


def pieplot(dict, colors=['#26547c', '#ef476f', '#ffd166', '#06d6a0', '#226ce0', '#e2afde'],
            plot_title=None, show_plot=False, save_plot=False, save_filename=None, clear_plot=True):
    # ''26547c-ef476f-ffd166-06d6a0-fcfcfc'
    plt.pie(x=dict.values(), colors=colors, autopct='%1.f%%', pctdistance=1.1, labeldistance=1.25)
    # plt.pie(x=dict.values(), colors=colors, autopct='%1.f%%', pctdistance=1.1, labeldistance=1.25,
    #         wedgeprops={'linewidth': 0, "edgecolor": '0.33'})
    plt.title(plot_title)
    plt.legend(dict.keys(), bbox_to_anchor=(0.85, 0.9), loc="upper left")
    if show_plot:
        plt.show()
    if save_plot:
        plt.savefig(plot_dir + save_filename, bbox_inches='tight')
    if clear_plot:
        plt.clf()


def stemplot(x, y, x_title=None, y_title=None, plot_title=None, show_plt=False, save_plot=False, save_filename=None):
    plt.stem(x, y, linefmt=":", markerfmt=None, basefmt=None)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(plot_title)
    if show_plt:
        plt.show()
    if save_plot:
        plt.savefig('plots/' + save_filename, bbox_inches='tight')
    plt.clf()


def autocorr(x):
    # calculates autocorrelation function of x
    N = len(x)
    norm = (x - np.mean(x))
    result = np.correlate(norm, norm, mode='full')
    n = result.size
    acorr = result[n // 2:] / (x.var() * np.arange(N, 0, -1))
    return acorr


def autocorr_plot(x, x_title='lag', y_title='autocorrelation', plot_title=None, show_plt=False, save_plot=False,
                  save_filename=None):
    plt.figure()
    autocorrelation_plot(x)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(plot_title)
    if show_plt:
        plt.show()
    if save_plot:
        plt.savefig('plots/' + save_filename, bbox_inches='tight')
    plt.clf()


def scatter_plot(x, y, marker='x', xlabel=None, ylabel=None,
                 plot_title=None, show_plt=False, save_plot=False, save_file_name=None, clear_plot=True):
    plt.scatter(x, y, marker=marker)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_title)
    if show_plt:
        plt.show()
    if save_plot:
        plt.savefig(save_file_name, bbox_inches='tight')
    if clear_plot:
        plt.clf()


def scatter_plot_multiple(list_of_xs, list_of_ys, xlabel=None, ylabel=None, series_legends=None, marker='x',
                          plot_title=None, show_plt=False, save_plot=False, save_file_name=None, clear_plot=True):
    for x, y in zip(list_of_xs, list_of_ys):
        scatter_plot(x, y, xlabel=None, ylabel=None,
                     marker=marker, plot_title=None, show_plt=show_plt, save_plot=save_plot,
                     save_file_name=save_file_name, clear_plot=clear_plot)
    plt.title(plot_title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(series_legends)
    if show_plt:
        plt.show()
    if save_plot:
        plt.savefig(save_file_name, bbox_inches='tight')
    if clear_plot:
        plt.clf()
