import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_analyze_data(path):
    df = pd.read_csv(path)
    stats = df.describe(include='all')

    plots = []

    fig1, ax1 = plt.subplots()
    df.groupby('Date')['Sales'].sum().plot(ax=ax1, title="Sales Over Time")
    plots.append(fig1)

    fig2, ax2 = plt.subplots()
    df.groupby('Product')['Sales'].sum().plot(kind='bar', ax=ax2, title="Sales by Product")
    plots.append(fig2)

    fig3, ax3 = plt.subplots()
    df.groupby('Region')['Sales'].sum().plot(kind='bar', ax=ax3, title="Sales by Region")
    plots.append(fig3)

    return df, stats, plots
