import pandas as pd

list_2d = [['a1', 'a2', 'a3'], ['b1', 'b2', 'b3']]
df = pd.DataFrame(list_2d, columns=['V1', 'V2', 'V3'])
print(df)

df.columns=['Vec1', 'Vec2', 'Vec3']
print(df)
