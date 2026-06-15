# read version from installed package
from importlib.metadata import version
__version__ = version("my_krml_24999690")

from .data.loaders import load_data_at2, get_experiment_files, load_data_at3, summarize_dataframe
from .data.exploration import get_shapes, check_duplicates, check_nulls, check_null_columns, check_duplicates_df, comprehensive_report
from .data.canvas_downloader import download_canvas_courses
from .features.engineering import add_cyclical_time_features, split_time_series, split_data, add_lags_stats_and_marketcap_changes, pop_target
from .features.preprocessing import standardize_train, standardize_test, drop_columns, remove_duplicates, create_dummies_train, create_dummies_test, create_dummies, DFStandardScaler, DFDummyEncoder, balance_classes
from .models.training import train_classifier, train_regressor, cross_validate_model, tune_hyperparameters
from .models.performance import evaluate_classification, evaluate_regression, print_metrics, compare_metrics, kaggle_submission, plot_feature_importance, plot_confusion_matrix, plot_roc_curve
from .models.tracking import ExperimentTracker
from .visualization.eda import plot_distribution, plot_barchart, plot_categorical, plot_boxplot, plot_scatter, plot_correlation_heatmap, plot_pairplot

__all__ = [
    "load_data_at2", "get_experiment_files", "load_data_at3", "summarize_dataframe",
    "get_shapes", "check_duplicates", "check_nulls", "check_null_columns", "check_duplicates_df", "comprehensive_report",
    "download_canvas_courses",
    "add_cyclical_time_features", "split_time_series", "split_data", "add_lags_stats_and_marketcap_changes", "pop_target",
    "standardize_train", "standardize_test", "drop_columns", "remove_duplicates", "create_dummies_train", "create_dummies_test", "create_dummies",
    "DFStandardScaler", "DFDummyEncoder", "balance_classes",
    "train_classifier", "train_regressor", "cross_validate_model", "tune_hyperparameters",
    "evaluate_classification", "evaluate_regression", "print_metrics", "compare_metrics", "kaggle_submission",
    "plot_feature_importance", "plot_confusion_matrix", "plot_roc_curve",
    "ExperimentTracker",
    "plot_distribution", "plot_barchart", "plot_categorical", "plot_boxplot", "plot_scatter", "plot_correlation_heatmap", "plot_pairplot"
]
