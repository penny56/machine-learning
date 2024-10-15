import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns


df = pd.read_csv("nigerian-songs.csv")
print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.describe())

