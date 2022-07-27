import pandas as pd
import numpy as np
import optparse as op

import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline
sns.set_style('whitegrid')

df_outpath = '/mnt/mcfiles/cpsolorzano/projects/leon_2020_braix-baseline/output/images/'
df_filepath = '/mnt/mcfiles/cpsolorzano/projects/leon_2020_braix-baseline/data/images_index_v0.2.csv'
dcm_filepath = '/mnt/mcfiles/cpsolorzano/projects/leon_2020_braix-baseline/data/dicoms_collated_all.csv'
img_path = '/mnt/beegfs/mcprojects/braix/data/admani1-transformed-v0.2/release/'

# set seed for reproducibility
np.random.seed(0) 

def print_basic(df):
    # get a small subset of the dataset (only columns required for specific application)
    cols = ['ImageId', 'ImageMainOutcome', 'ImagePhotometricInterpretation', 'ImageWidth', 'ImageHeight']
    subset_df = df.loc[:, cols]

    # look at the first five rows of the df file.
    print(subset_df.head())

    # get the number of missing data points per column
    missing_values_count = subset_df.isnull().sum()

    # look at the # of missing points in the first ten columns
    print(missing_values_count[0:10])

    # how many total missing values do we have?
    total_cells = np.product(subset_df.shape)
    total_missing = missing_values_count.sum()

    # percent of data that is missing
    percent_missing = (total_missing / total_cells) * 100
    print('percent_missing = ', percent_missing)

    # remove all the rows that contain a missing value
    subset_df.dropna()

    # remove all columns with at least one missing value
    columns_with_na_dropped = subset_df.dropna(axis=1)
    print('columns_with_na_dropped = ', columns_with_na_dropped.head())

    # just how much data did we lose?
    print("Columns in original subset: %d \n" % subset_df.shape[1])
    print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])

def optionFlags():
    parser = op.OptionParser("%program [options] headerfile datafile dataXMLfile")
    anGroup = op.OptionGroup(parser, 'anParams')

    parser.add_option("-f", "--function",
                      help="Defines the function to execute",
                      type="str", dest="fun",
                      default='print_basic')

    parser.add_option_group(anGroup)
    (opts, args) = parser.parse_args()
    return opts

if __name__ == "__main__":
    params = optionFlags()

    # read in all our data
    df_full = pd.read_csv(df_filepath, index_col='ImageId')

    # set seed for reproducibility
    np.random.seed(456)

    if params.fun == 'print_basic':
        print_basic(df_full)
    elif params.fun == 'plot_multivariable':
        plot_multivariable(df_full)