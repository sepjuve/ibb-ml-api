from loguru import logger
import numpy as np
import pandas as pd
import pickle, os

class TicketGroupServices(object):    
    text = ""
    arrText = []

    def __init__(self):                
        self._load_local_model()


    def _load_local_model(self):
        logger.debug("Load model..")
        self.model = pickle.load(open("ml_models/ticket_group.pkl", "rb"))


    def predict_once(self):
        logger.debug("Predicting..")
        testing = {"Text":[self.text]}
        X_pred = pd.DataFrame(testing)
        proba = self.model.predict_proba(X_pred.Text)

        prediksi = [self.model.classes_[np.argmax(x)] for x in proba]
        accuracy = [np.max(x) for x in proba]
        
        result = { 
            "text" : self.text,
            "class" : prediksi[0],
            "accuracy" : accuracy[0]
        }

        return result
    
    
    def predict_multi(self):
        logger.debug("Predicting..")
        testing = {"Text":self.arrText}
        X_pred = pd.DataFrame(testing)
        proba = self.model.predict_proba(X_pred.Text)
        X_pred["Prediksi"] = [self.model.classes_[np.argmax(x)] for x in proba]
        X_pred["Accuracy"] = [np.max(x) for x in proba]
        X_pred.to_json

        return X_pred
    