import pandas as pd
import numpy as np


#load data
df1=pd.read_csv('./datasets/cars.csv')
print(df1)

##creat dummis
dummies=pd.get_dummies(df1)
s2 = dummies.stack()
print(dummies)

#return dummies data to orginal data
orginal_df=pd.DataFrame({})
for i in dummies.columns[3:]:
    
    car_name=pd.DataFrame({'car':[i[10:]]})
    fitures=dummies[dummies[i]==1]
    fitures=fitures[fitures.columns[:3]]
    orginal_df1= car_name.join(fitures, how='right').fillna(i[10:])
    orginal_df=orginal_df.append(orginal_df1, ignore_index=True)


print(orginal_df)