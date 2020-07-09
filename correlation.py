import numpy as np
import myfuncs as my

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

corr_threshold = 0.75
corr, pvalues = my.corr_with_significance(df=data)
msk = np.ma.make_mask(np.tril(np.ones(corr.shape)))  # mask to remove the lower triangle of the correlation matrix
corr_msk = corr.mask(msk)  # apply mask
corr_sorted = corr_msk.unstack().sort_values(kind="quicksort") # sort correlation coefficients
corr_strong = corr_sorted[(abs(corr_sorted) > corr_threshold) & (corr_sorted < 1)]  # remove weak correlations


