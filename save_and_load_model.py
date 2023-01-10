import imp
import pickle
import joblib


# load the model by pickle
loaded_model = pickle.load(open('model_save_pickle.sav', 'rb'))
print("pickle=====",loaded_model.predict([[2,9,6],[12,10,10]]))


# save model by joblib
joblib.dump(loaded_model, 'model_save_joblib.sav')
 
 
# load the model by joblib
loaded_model = joblib.load('model_save_joblib.sav')
print('j===',loaded_model.predict([[2,9,6],[12,10,10]]))
