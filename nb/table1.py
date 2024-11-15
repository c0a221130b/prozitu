import pandas as pd

list_1d = ['a1', 'a2', 'a3']
df = pd.DataFrame(list_1d)
print(df)

list_2d = [['a1', 'a2', 'a3'], ['b1', 'b2', 'b3']]
df = pd.DataFrame(list_2d)
print(df)