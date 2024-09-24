import tensorflow as tf
import numpy as np

from tensorflow.keras.models import load_model

class KeyPointClassifier(object):
    def __init__(
        self,
        model_path=(r'C:\Users\aliha\OneDrive\Desktop\Hand_pred\keypoint_classifier\model.keras'),
        ):
        self.model=load_model(model_path)
    
        
    def __call__(
        self,
        data
        ):
        self.data=data
        self.predict=self.model.predict(data)
        self.best_prediction = np.argmax(self.predict, axis=-1)
        print(self.predict)
        return self.best_prediction
           
    
    
   
