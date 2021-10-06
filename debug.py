import pandas as pd

pd.set_option('display.max_colwidth', 200)
pd.set_option('display.max_rows', 500)
pd.set_option('display.expand_frame_repr', False)

data = pd.read_pickle("data.pkl")
data = data.dropna()
data = data[(data != 0).all(1)]
data = data.iloc[:,
       data.columns.get_level_values(1) == "close"]


print(data)

num = 0
for i in data.columns:
   num += data[i][data[i]==0].count()
print(num)
