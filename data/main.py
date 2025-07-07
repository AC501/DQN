from data.transferEntropy import *
import pandas as pd

xs = [0, 1, 1, 1, 1, 0, 0, 0, 0]
ys = [0, 0, 1, 1, 1, 1, 0, 0, 0]

a = transfer_entropy(xs, ys, k=2)
print(a)

with open('t2.txt', 'r') as file:
    lines = file.readlines()
df = pd.DataFrame(lines)
df = df.iloc[800:1000]
# df.set_axis(['c1'], axis=1)
df = df[0].str.split(",",expand = True)
# print(df.shape[1])
# print(df.iloc[:,3])
for i in [1,2,3,4,5,6,7,8,9]:
    for j in [1,2,3,4,5,6,7,8,9]:
        # print(df.iloc[:,i])
        t = transfer_entropy(df.iloc[:,i],df.iloc[:,j],k=2)
        print(i,",",j,",",t)