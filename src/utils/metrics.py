import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.utils.logging_config import get_logger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)
logger = get_logger()

class ModelMetrics:
    def __init__(self):
        self.metrics_dict = {}
        
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc_roc': roc_auc_score(y_true, y_pred_proba)
        }        
        return metrics
        
    def add_model_metrics(self, model_name, metrics):
        self.metrics_dict[model_name] = metrics
        
    def compare_models(self):
        if not self.metrics_dict:
            return None
        return pd.DataFrame(self.metrics_dict).T
        
    def get_best_model(self):
        if not self.metrics_dict:
            return None, None
        results_df = self.compare_models()
        best_model = results_df['accuracy'].idxmax()
        best_score = results_df.loc[best_model, 'accuracy']
        return best_model, best_score