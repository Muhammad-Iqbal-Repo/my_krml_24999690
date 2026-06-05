import os
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, Union

def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """
    Helper function to recursively flatten a dictionary, prepending the prefix to keys.
    """
    flat = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flat.update(_flatten_dict(v, prefix=f"{key}_"))
        else:
            # For lists, sets, tuples, convert to string so they save cleanly in CSV
            if isinstance(v, (list, set, tuple)):
                flat[key] = str(v)
            else:
                flat[key] = v
    return flat

class ExperimentTracker:
    """
    A class to track ML experiments (train/val/test splits, CV, hyperparameters).
    It maintains an in-memory Pandas DataFrame and immediately appends new logs 
    to a CSV file on disk to prevent data loss.
    """
    def __init__(self, filepath: str = "experiments_log.csv"):
        """
        Initializes the ExperimentTracker. If the CSV file already exists,
        it loads the existing logs.
        
        Parameters:
        -----------
        filepath : str
            The path to the CSV file where experiments will be logged.
        """
        self.filepath = filepath
        if os.path.exists(self.filepath):
            try:
                self.df = pd.read_csv(self.filepath)
            except Exception as e:
                # If there's an error reading (e.g. empty file), initialize empty DataFrame
                self.df = pd.DataFrame()
        else:
            self.df = pd.DataFrame()

    def log_experiment(
        self,
        experiment_name: str,
        model_name: Union[str, Any],
        parameters: Union[Dict[str, Any], Any],
        metrics: Dict[str, Any],
        cv_score: Optional[Union[float, Dict[str, Any]]] = None,
        notes: str = ""
    ) -> pd.DataFrame:
        """
        Logs a new experiment run, updates the internal DataFrame, and appends to the CSV.
        
        Parameters:
        -----------
        experiment_name : str
            The name or identifier of the experiment setting/run.
        model_name : str or estimator object
            The name/type of the model or a scikit-learn estimator object.
        parameters : dict or estimator object
            Hyperparameters for the run or a scikit-learn estimator object (to auto-extract parameters).
        metrics : dict
            Evaluation metrics (e.g., train/val/test performance).
        cv_score : float or dict, optional
            Cross-validation score or dict of CV metrics.
        notes : str, optional
            Additional notes or comments about this run.
            
        Returns:
        --------
        pd.DataFrame
            The updated experiment log as a Pandas DataFrame.
        """
        # Auto-detect class name if model_name is a model object
        if not isinstance(model_name, str) and hasattr(model_name, "__class__"):
            model_name = model_name.__class__.__name__

        # Auto-extract hyperparameters if parameters is a model object with get_params()
        if not isinstance(parameters, dict) and hasattr(parameters, "get_params"):
            # deep=False is cleaner for logging constructor-level parameters
            raw_params = parameters.get_params(deep=False)
            try:
                # Try to instantiate a default version of the same estimator class to get its default parameters
                default_model = parameters.__class__()
                default_params = default_model.get_params(deep=False)
                # Keep only parameters that differ from the default
                parameters = {k: v for k, v in raw_params.items() if v != default_params.get(k)}
            except Exception:
                # Fallback to all parameters if default instantiation fails
                parameters = raw_params
        elif not isinstance(parameters, dict):
            # Fallback for non-dictionary/non-estimator objects
            parameters = {}

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        row = {
            "timestamp": timestamp,
            "experiment_name": experiment_name,
            "model_name": model_name,
        }
        
        # Flatten and merge parameters and metrics with prefixes
        flat_params = _flatten_dict(parameters, prefix="param_")
        flat_metrics = _flatten_dict(metrics, prefix="metric_")
        
        row.update(flat_params)
        row.update(flat_metrics)
        
        # Add CV score(s)
        if cv_score is not None:
            if isinstance(cv_score, dict):
                flat_cv = _flatten_dict(cv_score, prefix="cv_")
                row.update(flat_cv)
            else:
                row["cv_score"] = cv_score
                
        row["notes"] = notes
        
        # Convert to a single-row DataFrame
        new_row_df = pd.DataFrame([row])
        
        if self.df.empty:
            self.df = new_row_df
        else:
            # We use pd.concat to merge, alignment is automatic and missing columns are filled with NaN
            self.df = pd.concat([self.df, new_row_df], ignore_index=True)
            
        # Save to CSV
        try:
            # Check if directory exists, if not create it
            dir_name = os.path.dirname(self.filepath)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
                
            self.df.to_csv(self.filepath, index=False)
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to save experiment log to {self.filepath}: {e}")
            
        return self.df

    def get_log(self, sort_by: Optional[str] = None, ascending: bool = False) -> pd.DataFrame:
        """
        Returns the formatted experiment log.
        
        Parameters:
        -----------
        sort_by : str, optional
            Column name to sort the log by. E.g., 'metric_val_accuracy' or 'timestamp'.
        ascending : bool, default False
            Whether to sort in ascending order (default is False, so highest/newest is on top).
            
        Returns:
        --------
        pd.DataFrame
            The experiment log DataFrame.
        """
        if self.df.empty:
            return self.df.copy()
            
        df_to_return = self.df.copy()
        if sort_by is not None:
            if sort_by in df_to_return.columns:
                df_to_return = df_to_return.sort_values(by=sort_by, ascending=ascending)
            else:
                import warnings
                warnings.warn(f"Column '{sort_by}' not found in log. Available columns: {list(df_to_return.columns)}")
                
        return df_to_return

    def plot_comparison(self, metric: str, title: Optional[str] = None):
        """
        Plots a bar chart comparing different experiments on a specific metric.
        
        Parameters:
        -----------
        metric : str
            The metric column name (e.g. 'metric_val_accuracy' or 'metric_accuracy').
        title : str, optional
            Title for the plot.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if self.df.empty:
            print("No experiments logged yet.")
            return
            
        # Check if the requested metric exists in columns
        actual_metric = metric
        if metric not in self.df.columns and f"metric_{metric}" in self.df.columns:
            actual_metric = f"metric_{metric}"
            
        if actual_metric not in self.df.columns:
            print(f"Metric '{metric}' not found in log. Available columns: {list(self.df.columns)}")
            return
            
        # Clean/drop rows where metric is null
        plot_df = self.df.dropna(subset=[actual_metric]).copy()
        if plot_df.empty:
            print(f"No non-null data found for metric '{actual_metric}'.")
            return
            
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=plot_df,
            x=actual_metric,
            y="experiment_name",
            hue="model_name",
            dodge=False
        )
        
        plt.title(title or f"Comparison of Experiments by {actual_metric}")
        plt.xlabel(actual_metric)
        plt.ylabel("Experiment Name")
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def clear_log(self, confirm: bool = False):
        """
        Clears the experiment log both in-memory and on disk.
        
        Parameters:
        -----------
        confirm : bool, default False
            Must be set to True to confirm the deletion.
        """
        if not confirm:
            print("Please set confirm=True to clear the logs.")
            return
            
        self.df = pd.DataFrame()
        if os.path.exists(self.filepath):
            try:
                os.remove(self.filepath)
            except Exception as e:
                import warnings
                warnings.warn(f"Failed to remove CSV file at {self.filepath}: {e}")
        print("Experiment log cleared successfully.")
