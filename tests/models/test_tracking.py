import os
import pytest
import pandas as pd
from my_krml_24999690.models.tracking import ExperimentTracker

def test_experiment_tracker(tmp_path):
    # Use a temp directory for the log file to isolate tests
    log_file = os.path.join(tmp_path, "test_experiments.csv")
    
    # 1. Initialize tracker
    tracker = ExperimentTracker(filepath=log_file)
    assert tracker.df.empty
    
    # 2. Log an experiment
    params = {"learning_rate": 0.01, "max_depth": 5}
    metrics = {"train_acc": 0.95, "val_acc": 0.88}
    tracker.log_experiment(
        experiment_name="run_1",
        model_name="RandomForest",
        parameters=params,
        metrics=metrics,
        cv_score=0.85,
        notes="First run baseline"
    )
    
    # Check that CSV was written
    assert os.path.exists(log_file)
    
    # Check internal dataframe
    df = tracker.get_log()
    assert len(df) == 1
    assert df.loc[0, "experiment_name"] == "run_1"
    assert df.loc[0, "model_name"] == "RandomForest"
    assert df.loc[0, "param_learning_rate"] == 0.01
    assert df.loc[0, "metric_val_acc"] == 0.88
    assert df.loc[0, "cv_score"] == 0.85
    assert df.loc[0, "notes"] == "First run baseline"
    
    # 3. Log a second experiment with nested metrics and different params
    nested_metrics = {
        "train": {"loss": 0.1, "accuracy": 0.98},
        "val": {"loss": 0.3, "accuracy": 0.90}
    }
    tracker.log_experiment(
        experiment_name="run_2",
        model_name="XGBoost",
        parameters={"learning_rate": 0.1, "n_estimators": 200},
        metrics=nested_metrics,
        notes="Second run with XGBoost"
    )
    
    df2 = tracker.get_log()
    assert len(df2) == 2
    # Verify flattening of nested metrics
    assert df2.loc[1, "metric_train_loss"] == 0.1
    assert df2.loc[1, "metric_val_accuracy"] == 0.90
    # Missing columns for xgboost run in randomforest columns should be NaN (or missing)
    assert pd.isna(df2.loc[1, "param_max_depth"])
    
    # 4. Re-instantiate tracker to ensure it loads existing CSV
    new_tracker = ExperimentTracker(filepath=log_file)
    assert len(new_tracker.df) == 2
    assert "param_learning_rate" in new_tracker.df.columns
    
    # 5. Test sorting
    sorted_df = new_tracker.get_log(sort_by="metric_val_acc", ascending=False)
    # The first run had metric_val_acc = 0.88, second run had metric_val_accuracy = 0.90 but metric_val_acc is NaN
    # Let's sort by param_learning_rate: run_1 has 0.01, run_2 has 0.1
    sorted_df = new_tracker.get_log(sort_by="param_learning_rate", ascending=False)
    assert sorted_df.iloc[0]["experiment_name"] == "run_2"
    assert sorted_df.iloc[1]["experiment_name"] == "run_1"
    
    # 6. Test clearing log
    new_tracker.clear_log(confirm=True)
    assert new_tracker.df.empty
    assert not os.path.exists(log_file)

def test_experiment_tracker_with_model_object(tmp_path):
    from sklearn.tree import DecisionTreeClassifier
    
    log_file = os.path.join(tmp_path, "test_model_experiments.csv")
    tracker = ExperimentTracker(filepath=log_file)
    
    # Instantiate a model
    model = DecisionTreeClassifier(max_depth=4, min_samples_split=5, random_state=42)
    
    # Log the experiment, passing the model for both model_name and parameters
    tracker.log_experiment(
        experiment_name="dt_test",
        model_name=model,
        parameters=model,
        metrics={"accuracy": 0.85}
    )
    
    df = tracker.get_log()
    assert len(df) == 1
    assert df.loc[0, "model_name"] == "DecisionTreeClassifier"
    assert df.loc[0, "param_max_depth"] == 4
    assert df.loc[0, "param_min_samples_split"] == 5
    assert df.loc[0, "param_random_state"] == 42
    assert df.loc[0, "metric_accuracy"] == 0.85
