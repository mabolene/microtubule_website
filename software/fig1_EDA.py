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






#pull data path from microtubule_functions
data_path = microtubule_functions.data_path

#get full path name
fname = os.path.join(data_path, "gardner_mt_catastrophe_only_tubulin.csv")

#create dataframe
df_all = pd.read_csv(fname, na_values="*", skiprows=9)

#drop extra unneeded column
#df_all = df_all.drop('Unnamed: 5', axis=1)

#restructure for ECDF
df = df_all.melt()

#drop nans
df = df.dropna()

#define colors for plot
colors = ['#80b1d3', '#fc8d62', '#66c2a5', '#e78ac3', '#756bb1']

#plot ECDF
c = iqplot.ecdf(df, 
                q = 'value',
                cats='variable',
                palette=colors
               )

c.xaxis.axis_label = 'Time to catastrophe (s)'
c.title='Time to catastrophe for different tubulin concentrations'
bokeh.io.show(c)

#enable saving
bokeh.plotting.output_file(filename="Team18_fig1.html", title="EDA")
bokeh.plotting.save(c)