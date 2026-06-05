# read version from installed package
from importlib.metadata import version
__version__ = version("my_krml_24999690")

from .data.loaders import load_data_at2, get_experiment_files, load_data_at3, summarize_dataframe
from .data.exploration import get_shapes, drop_columns, check_duplicates, check_nulls, check_null_columns, check_duplicates_df
from .data.canvas_downloader import download_canvas_courses
from .features.engineering import add_cyclical_time_features, split_time_series, add_lags_stats_and_marketcap_changes, pop_target
from .features.preprocessing import standardize_train, standardize_test
from .models.training import at2_create_time_series_splits, at2_train_evaluate_classification_model, at2_prepare_training, train_evaluate_regression_model
from .models.performance import evaluate_model, print_auroc, kaggle_submission
from .visualization.eda import plot_class_distribution, boxplot_regression, plot_categorical_distribution_with_target, plot_numerical_with_target, plot_distribution_and_trend

__all__ = [
    "load_data_at2", "get_experiment_files", "load_data_at3", "summarize_dataframe",
    "get_shapes", "drop_columns", "check_duplicates", "check_nulls", "check_null_columns", "check_duplicates_df",
    "download_canvas_courses",
    "add_cyclical_time_features", "split_time_series", "add_lags_stats_and_marketcap_changes", "pop_target",
    "standardize_train", "standardize_test",
    "at2_create_time_series_splits", "at2_train_evaluate_classification_model", "at2_prepare_training", "train_evaluate_regression_model",
    "evaluate_model", "print_auroc", "kaggle_submission",
    "plot_class_distribution", "boxplot_regression", "plot_categorical_distribution_with_target", "plot_numerical_with_target", "plot_distribution_and_trend"
]