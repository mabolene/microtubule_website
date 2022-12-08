# Colab setup ------------------
import os, sys, subprocess
if "google.colab" in sys.modules:
    cmd = "pip install --upgrade iqplot bebi103 watermark"
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    data_path = "https://s3.amazonaws.com/bebi103.caltech.edu/data/"
else:
    pass
    #data_path = "../data/"
# ------------------------------


# Our main plotting package (must have explicit import of submodules)
import bokeh.io
import bokeh.plotting

# Enable viewing Bokeh plots in the notebook
bokeh.io.output_notebook()


import numpy as np
import pandas as pd
import pandas
import math

import bebi103
import iqplot

from matplotlib import pyplot as plt
import tqdm

import scipy.optimize
import scipy.stats as st
import scipy.special
import scipy

import microtubule_functions
from microtubule_functions import gamma_mle_computer, two_beta_mle_computer, two_beta_qq, gamma_qq
#############################################################

#fetch data
data_path = microtubule_functions.data_path
fname = os.path.join(data_path, "gardner_mt_catastrophe_only_tubulin.csv")
df_all = pd.read_csv(fname, na_values="*", skiprows=9)

#restructure data
df = df_all.melt()
df = df.dropna()

#extract time lists from dataframe
time12 = df.loc[df['variable']== '12 uM', 'value'].values
time7 = df.loc[df['variable']== '7 uM', 'value'].values
time9 = df.loc[df['variable']== '9 uM', 'value'].values
time10 = df.loc[df['variable']== '10 uM', 'value'].values
time14 = df.loc[df['variable']== '14 uM', 'value'].values

#compute MLEs for gamma
mle_7_g = gamma_mle_computer(time7, '7 uM')
mle_9_g = gamma_mle_computer(time9, '9 uM')
mle_10_g = gamma_mle_computer(time10, '10 uM')
mle_12_g = gamma_mle_computer(time12, '12 uM')
mle_14_g = gamma_mle_computer(time14, '14 uM')
bigl_g = [['tubulin uM', 'alpha', 'beta'], mle_7_g[:3], mle_9_g[:3], mle_10_g[:3], mle_12_g[:3], mle_14_g[:3]]

#compute MLEs for 2-beta
mle_7_2b = two_beta_mle_computer(time7, '7 uM')
mle_9_2b = two_beta_mle_computer(time9, '9 uM')
mle_10_2b = two_beta_mle_computer(time10, '10 uM')
mle_12_2b = two_beta_mle_computer(time12, '12 uM')
mle_14_2b = two_beta_mle_computer(time14, '14 uM')

bigl_2b = bigl_2b = [['tubulin uM', 'beta1', 'beta2', 'beta2 % infinity'], mle_7_2b[:4], mle_9_2b[:4], mle_10_2b[:4], mle_12_2b[:4], mle_14_2b[:4]]

#make big list for times with matching indices to bigl_x lists
bigtime = [0, time7, time9, time10, time12, time14]

row_list = []
for i in range(len(bigl_2b)):
    if i ==0:
        pass
    else:
        bb = two_beta_qq(bigl_2b[i][1], bigl_2b[i][2], bigtime[i], str(bigl_2b[i][0]), 'contour')
        bbb = two_beta_qq(bigl_2b[i][1], bigl_2b[i][2], bigtime[i], str(bigl_2b[i][0]), 'diff')

        gg = gamma_qq(bigl_g[i][1], bigl_g[i][2], bigtime[i], str(bigl_g[i][0]), 'contour')
        ggg = gamma_qq(bigl_g[i][1], bigl_g[i][2], bigtime[i], str(bigl_g[i][0]), 'diff')

        row1 = bokeh.layouts.row(
            bb,
            bokeh.models.Spacer(width=20),
            gg,
            bokeh.models.Spacer(width=20),
            bbb,
            bokeh.models.Spacer(width=20),
            ggg)
        row_list.append(row1)



dashboard = bokeh.layouts.column(row_list[0], bokeh.models.Spacer(width=30),
                                 row_list[1], bokeh.models.Spacer(width=30),
                                 row_list[2], bokeh.models.Spacer(width=30),
                                 row_list[3], bokeh.models.Spacer(width=30),
                                 row_list[4], bokeh.models.Spacer(width=30), 
                                )
bokeh.io.show(dashboard)

#enable saving of figure
bokeh.plotting.output_file(filename="Team18_fig2.html", title="Graphical model comparison")
bokeh.plotting.save(dashboard)