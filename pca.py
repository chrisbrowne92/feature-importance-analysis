
import pandas as pd
import matplotlib.pyplot as plt
import myfuncs as my
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

## PCA
# Separating out the features, don't include consumption
x = data.drop('Consumption', axis=1).values
# Standardizing the features
x = StandardScaler().fit_transform(x)

n_components = 100  # set number of components
pca = PCA(n_components=n_components)  # create PCA model
x_in_new_base = pca.fit_transform(x)  # fit model from data
components = pca.components_  # get the components
variance = pca.explained_variance_ratio_  # get explained variance of each component
principal_component = pd.Series(components[0, :], index=data.drop('Consumption', axis=1).columns)
plot_pca = False
plot_variance = True
if plot_pca:
    data_2d = pd.DataFrame(data=x_in_new_base, columns=['pc1', 'pc2'], index=data.index)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    ax.scatter(data_2d.loc[:, 'pc1']
               , data_2d.loc[:, 'pc2']
               , c='b'
               , s=50)
    ax.grid()
    plt.show()
    save_name = 'pca.png'
    plt.savefig(plt_dir + save_name, bbox_inches='tight')
    plt.clf()
if plot_variance:
    labels = [str(i+1) for i in range(n_components)]
    explained_variance = pd.Series(data=variance, index=labels)
    # fig = plt.figure(figsize=(8, 8))
    my.barplot_single(explained_variance,
                      plot_title='num components = ' + str(n_components),
                      saveFileName='pca_explained_variance--num_components_' + str(n_components) + '.png',
                      savePlot=True,
                      skip_ticks=5,
                      xlabel='Principal component',
                      ylabel='Explained variance ratio',
                      clearPlot=True)
