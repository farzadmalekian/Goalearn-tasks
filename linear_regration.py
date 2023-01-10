import pandas as pd
from sklearn import linear_model
import pickle



# load data
df = pd.read_csv('./datasets/salery.csv')
df=df.rename(columns={'test_score(out of 10)':'test_score' , 'interview_score(out of 10)': 'interview_score','salary($)':'salary'})
df=df.replace({"experience":{'two':2.,'three':3, 'five':5, 'seven':7,'ten':10, 'eleven':11}})

#handele missing data
median=math.floor(df.experience.median())
df.experience = df.experience.fillna(math.floor(df.experience.median()))
df.test_score=df.test_score.fillna(math.floor(df.test_score.median()))

#learn model
reg = linear_model.LinearRegression()
reg.fit(df[['experience','test_score','interview_score']],df.salary)


#save result
with open('predicted_salaries.txt',"w") as file:
    file.write(f"for 2 yr experience, 9 test score, 6 interview score salery is :{reg.predict([[2,6,9]])} \nfor 12 yr experience, 10 test score, 10 interview score salery is :{reg.predict([[12,10,10]])}")

    
#save model    
pickle.dump(reg, open('model_save_pickle.sav', 'wb'))