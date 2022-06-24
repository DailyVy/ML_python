import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


df = pd.read_csv("Iris.csv")
# print(df.head())

df.set_index("Id", inplace=True)
print(df.head())

sns.pairplot(df, hue='Species')
plt.show()
