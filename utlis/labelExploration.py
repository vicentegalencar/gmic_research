import pandas as pd
from collections import Counter
import numpy as np
import csv
import optparse as op

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import sys

font = {'size'   : 20}
matplotlib.rc('font', **font)
matplotlib.rcParams['figure.dpi'] = 300

sns.set_style('white') # whitegrid, dark, white, ticks,

df_outpath = '/mnt/mcfiles/cpsolorzano/projects/leon_2020_braix-baseline/output/images/'
df_filepath = '/mnt/mcfiles/cpsolorzano/projects/leon_2020_braix-baseline/data/images_index_v0.2.csv'
dcm_filepath = '/mnt/mcfiles/cpsolorzano/projects/leon_2020_braix-baseline/data/dicoms_collated_all.csv'
img_path = '/mnt/beegfs/mcprojects/braix/data/admani1-transformed-v0.2/release/'

def label_histo(df, output_path):

    df['ImageResolution'] = df['ImageWidth'] *  df['ImageHeight']
    df['ImageAspectRatio'] = df['ImageWidth'] / df['ImageHeight']

    ### Distribution histograms (whole dataset) ###
    plt.figure()
    sns.countplot(df['ImageMainOutcome'])
    plt.title('Image Main Outcome (Full database)')
    plt.ylabel('Frequency')
    plt.savefig(output_path + 'HistoFull_ImageMainOutcome')

    plt.figure()
    sns.distplot(a=df['ImageAspectRatio'], kde=False, hist_kws={'edgecolor':'black'})
    plt.title('Image Squareness (Full database)')
    plt.yscale('log')
    plt.ylabel('Frequency')
    plt.savefig(output_path + 'HistoFull_ImageAspectRatio')

    plt.figure()
    sns.countplot(df['ImagePhotometricInterpretation'])
    plt.title('Image Photometric Interpretation (Full database)')
    plt.ylabel('Frequency')
    plt.savefig(output_path + 'HistoFull_ImagePhotometricInterpretation')

    f, (ax_top, ax_bottom) = plt.subplots(ncols=1, nrows=2, sharex=True, gridspec_kw={'hspace': 0.1})

    # zoom-in / limit the view to different portions of the data
    ax_top.set_ylim(bottom=2000, top=45000)  # most of the data
    ax_bottom.set_ylim(0, 800)  # outliers only

    # hide the spines between ax and ax2
    sns.despine(ax=ax_bottom)
    sns.despine(ax=ax_top, bottom=True)

    bins = 20
    g0 = sns.distplot(a=df['ImageWidth'], kde=False, bins=bins, label='Image width', ax=ax_top, hist_kws={'edgecolor':'black'})
    sns.distplot(a=df['ImageHeight'], kde=False, bins=bins, label='Image height', ax=ax_top, hist_kws={'edgecolor':'black'})
    g1 = sns.distplot(a=df['ImageWidth'], kde=False, bins=bins, label='Image width', ax=ax_bottom, hist_kws={'edgecolor':'black'})
    sns.distplot(a=df['ImageHeight'], kde=False, bins=bins, label='Image height', ax=ax_bottom, hist_kws={'edgecolor':'black'})

    g0.set(xlabel=None)
    g1.set(xlabel=None)

    d = .01  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal

    kwargs.update(transform=ax_bottom.transAxes)  # switch to the bottom axes
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal

    plt.suptitle('Image Width and Height (Full database)')
    ax_top.legend()
    plt.ylabel('Frequency')
    plt.savefig(output_path + 'HistoFull_ImageWidthHeight')

    plt.figure()
    sns.distplot(a=df['ImageResolution'], kde=False, hist_kws={'edgecolor':'black'})
    plt.title('Image Resolution (Full database)')
    plt.ylabel('Frequency')
    plt.savefig(output_path + 'HistoFull_ImageResolution')

    ### Distribution histograms (cancer only) ###
    plt.figure()
    sns.distplot(a=df.loc[df.ImageMainOutcome == 1, 'ImageAspectRatio'], kde=False, hist_kws={'edgecolor':'black'})
    plt.title('Image Squareness (Cancer)')
    plt.yscale('log')
    plt.ylabel('Frequency')
    plt.savefig(output_path + 'HistoCancer_ImageAspectRatio')

    plt.figure()
    sns.countplot(df.loc[df.ImageMainOutcome == 1, 'ImagePhotometricInterpretation'])
    plt.title('Image Photometric Interpretation (Cancer)')
    plt.ylabel('Frequency')
    plt.savefig(output_path + 'HistoCancer_ImagePhotometricInterpretation')

    plt.figure()
    sns.distplot(a=df.loc[df.ImageMainOutcome == 1, 'ImageWidth'], kde=False, hist_kws={'edgecolor':'black'}, label='Image width')
    sns.distplot(a=df.loc[df.ImageMainOutcome == 1, 'ImageHeight'], kde=False, hist_kws={'edgecolor':'black'}, label='Image height')
    plt.title('Image Width and Height (Cancer)')
    plt.legend()
    plt.ylabel('Frequency')
    plt.savefig(output_path + 'HistoCancer_ImageWidthHeight')

    plt.figure()
    sns.distplot(a=df.loc[df.ImageMainOutcome == 1, 'ImageResolution'], hist=True, kde=False, hist_kws={'edgecolor':'black'})
    plt.title('Image Resolution (Cancer)')
    plt.ylabel('Frequency')
    plt.savefig(output_path + 'HistoCancer_ImageResolution')

    bins = 20
    plt.figure()
    sns.distplot(a=df['ImageAspectRatio'], kde=False, bins=bins, hist_kws={'edgecolor':'black'}, label='Full database')
    sns.distplot(a=df.loc[df.ImageMainOutcome==1, 'ImageAspectRatio'], kde=False, bins=bins, hist_kws={'edgecolor':'black'}, label='Cancer')
    plt.title('Image Squareness')
    plt.yscale('log')
    plt.legend()
    plt.ylabel('Frequency')
    plt.savefig(output_path + 'Histo_ImageAspectRatio')

    print('Histograms saved to path %s'%output_path)

def count_unique(df, output_file):
    self.output_file = output_file

    def count_cols(col, writer):
        val_count = col.value_counts()
        writer.writerow([col.name])
        writer.writerow(val_count.index)
        writer.writerow(val_count.values)
        writer.writerow('')
        print('Count of unique values in column {} saved to {}'.format(col.name, self.output_file))
    f = open(self.output_file, "w")
    writer = csv.writer(f)
    df[['ClientAgeGroup','ImageMainOutcome','ImageSubOutcome','ImageOriginalBitDepth','ImageWidth',
       'ImageHeight','ImageWhiteBorder', 'ImageError']].apply(lambda x:count_cols(x,writer), axis=0)
    f.close()

def plot_categoricals(df, x, y=None, hue=None, order=None, edit_legend=False, legend_title=None, legend_labels=None, yscale='log'):
    fig, ax = plt.subplots()

    if y == None:
        kind = 'count'
        ylabel = 'Number of images'
        g = sns.catplot(data=df,
                        x=x,
                        y=y,
                        kind=kind,
                        hue=hue,
                        palette="pastel",
                        order=order,
                        legend_out=False,
                        height=15, aspect=1.2,
                        )
    else:
        kind = 'violin'
        ylabel = y
        g = sns.catplot(data=df,
                        x=x,
                        y=y,
                        kind=kind,
                        hue=hue,
                        palette="pastel",
                        order=order,
                        legend_out=False,
                        height=15, aspect=1.2,
                        split=True,
                        )

    g.set(yscale=yscale, ylabel=ylabel)
    g.set_xticklabels(rotation=30, ha='right')
    if edit_legend:
        leg = g.axes.flat[0].get_legend()
        leg.set_title(legend_title)
        for t, l in zip(leg.texts, legend_labels): t.set_text(l)
    # hue = hue if y==None else y
    g.fig.suptitle('{} vs {}'.format(x, hue))
    if y == None:
        plt.savefig(df_outpath + '{}_{}'.format(x, hue), bbox_inches='tight')
    else:
        plt.savefig(df_outpath + '{}_{}_{}'.format(x, y, hue), bbox_inches='tight')
    plt.close()

def plot_multivariable(df):
    df['ImageResolution'] = round(df['ImageWidth'] * df['ImageHeight'] / 1000000)
    df['ImageAspectRatio'] = round(df['ImageHeight'] / df['ImageWidth'], 2)

    plot_categorical = True
    if plot_categorical:
        print('Plotting categorical data (bar plot) ...')
        plot_categoricals(df, x='ClientAgeGroup', hue='ImageMainOutcome', order=["40-49", "50-59", "60-69", "70-74", "75+"],
                          edit_legend=True, legend_title='Cancer outcome', legend_labels=['Control', 'Cancer'])
        plot_categoricals(df, x='ClientAgeGroup', hue='ImageSubOutcome', order=["40-49", "50-59", "60-69", "70-74", "75+"],
                          edit_legend=True, legend_title='Cancer outcome',
                          legend_labels=['Control', 'Cancer', 'Interval cancer', 'Benign', 'No significant abnormality'])
        plot_categoricals(df, x='ImagePhotometricInterpretation', hue='ImageMainOutcome', order=["MONOCHROME1", "MONOCHROME2"])
        plot_categoricals(df, x='ImageOriginalBitDepth', hue='ImageMainOutcome')
        plot_categoricals(df, x='ImageResolution', hue='ImageMainOutcome', yscale='linear')
        plot_categoricals(df, x='ImageAspectRatio', hue='ImageMainOutcome')
        plot_categoricals(df, x='ImageModality', hue='ImageMainOutcome')
        plot_categoricals(df, x='ClientAgeGroup', hue='TumourSizeGroup', order=["40-49", "50-59", "60-69", "70-74", "75+"])
        plot_categoricals(df, x='ImageManufacturer', hue='ImageMainOutcome')
        plot_categoricals(df, x='ImageManufacturer', hue='ImageModality')
        plot_categoricals(df, x='ImageManufacturerModelName', hue='ImageMainOutcome')
        plot_categoricals(df, x='ImageManufacturer', hue='ImagePhotometricInterpretation')
        plot_categoricals(df, x='ImageModality', hue='ImagePhotometricInterpretation')
        plot_categoricals(df, x='ClientAgeGroup', hue='ImageIsBreastImplantPresent', order=["40-49", "50-59", "60-69", "70-74", "75+"],
                          edit_legend=True, legend_title='Breast implant', legend_labels=['No present', 'Present'])
        plot_categoricals(df, x='ImageManufacturer', hue='ImageOriginalBitDepth')

    plot_fuji = False
    if plot_fuji:
        df_fuji = df.loc[(df['ImageManufacturer'] == 'FUJIFILM Corporation') & (df['ImageOriginalBitDepth'] == 12)]
        # plot_categoricals(df_fuji, x='ImageManufacturer', hue='ImageModality')

        kind = 'count'
        x='ImageModality'
        hue='ImageOriginalBitDepth'
        g = sns.catplot(data=df_fuji,
                        x=x,
                        kind=kind,
                        hue=hue,
                        palette='pastel',
                        legend_out=False,
                        height=15, aspect=1.2,
                        )
        plt.savefig(df_outpath + 'Fujifilm_{}_{}'.format(x, hue), bbox_inches='tight')
        plt.close()

    ### Violin with three variables (split by hue variable) ###
    plot_categoricals(df, x='ImagePhotometricInterpretation', y='ImageResolution', hue='ImageMainOutcome', yscale='linear')
    plot_categoricals(df, x='ImageManufacturer', y='ImageOriginalBitDepth', hue='ImageMainOutcome', yscale='linear')

    cols = ['ImagePhotometricInterpretation', 'ImageManufacturer', 'ImageOriginalBitDepth', 'ImageModality']
    fig, ax = plt.subplots(ncols=1, nrows=4, figsize=(40, 40))
    for i, x in enumerate(cols):
        g = sns.violinplot(data=df,
                        x=x,
                        y='ImageResolution',
                        hue='ImageMainOutcome',
                        palette="pastel",
                        legend_out=False,
                        height=15, aspect=1.2,
                        split=True,
                        ax=ax[i],
                        )
    plt.savefig(df_outpath + 'Several_ImageMainOutcome', bbox_inches='tight')
    plt.close()

    # plot_categoricals(df, x='ImageViewPosition', y='ClientAgeGroup', hue='ImageMainOutcome')
    # print('Plotting categorical age data (heatmap) ...')
    # fig, ax = plt.subplots()
    # subset_cancer_control = pd.pivot_table(df, values='ImageMainOutcome', index='ClientAgeGroup', columns=['ImageMainOutcome'],
    #                                        aggfunc=lambda x: x.value_counts().count())
    # print(subset_cancer_control.head())
    # g = sns.heatmap(subset_cancer_control, annot=True, linewidths=.3, ax=ax, cmap='RdPu')
    # plt.savefig(df_outpath + 'test', bbox_inches='tight')

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
    np.random.seed(453)

    if params.fun == 'labels_histogram':
        label_histo(df_full, sys.argv[3])
    elif params.fun == 'plot_multivariable':
        plot_multivariable(df_full)
    else:
        count_unique(df_full, sys.argv[3])
    print('Done')