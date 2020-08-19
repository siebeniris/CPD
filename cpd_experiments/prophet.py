import os
import logging
import warnings

import rootpath
import pandas as pd
from fbprophet.plot import add_changepoints_to_plot
from fbprophet import Prophet
from matplotlib import pyplot as plt
import numpy as np

from prepare_cpd import get_cpd_df, get_info_list, select_reviews


logging.getLogger('fbprophet').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def predict_(df):
    df = df.rename(columns={"date": "DS", "sentiment": "Y"})
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    fig = m.plot(forecast)
    for cp in m.changepoints:
        plt.axvline(cp, c="red", ls="--", lw=2)



if __name__ == '__main__':
    root_dir = rootpath.detect()
    filepath = os.path.join(root_dir, 'data', 'cpd_aspects', '6#ff018f51-ef0f-48d1-8c8f-636b4df3c1ff')

    cpd_df, df = get_cpd_df(filepath, "room")
    print(cpd_df)

