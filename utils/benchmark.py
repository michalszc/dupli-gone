import time
from typing import Any, Dict, List

import numpy as np
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Benchmark:
    '''
    A class for benchmarking classification models.

    Args:
        model (Any): A model object that has a `predict` method.
    '''

    def __init__(self, model: Any):
        self.model = model

    def evaluate(self, X: List[str], y_true: np.ndarray, verbose: bool = False) -> Dict[str, float]:
        '''
        Calculates classification metrics and prediction time.

        Args:
            X: Input features.
            y_true: True labels.
            verbose: If True, prints the summary of metrics.

        Returns:
            dict: A dictionary containing metrics and prediction time.
        '''
        start_time = time.time()
        y_pred = self.model.predict(X)
        end_time = time.time()

        sample_weights = compute_sample_weight(class_weight='balanced', y=y_true)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred, sample_weight=sample_weights),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0, sample_weight=sample_weights),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0, sample_weight=sample_weights),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0, sample_weight=sample_weights),
            'prediction_time_sec': end_time - start_time,
            'samples': len(y_true),
            'duplicates': sum(y_true),
        }

        if verbose:
            print('Summary:')
            print(f"{'Metric':<20}{'Value':>15}")
            print('-' * 35)
            for metric, value in metrics.items():
                print(f'{metric.capitalize():<20}{value:>15.5f}')

        return metrics
