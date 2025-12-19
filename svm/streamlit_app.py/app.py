import pickle
with open("Logistic_Regression.pkl","wb") as f:
    pickle.dump(Logr,f)
from google.colab import files
files.download('Logistic_Regression.pkl')
