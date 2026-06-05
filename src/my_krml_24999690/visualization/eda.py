import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde

def plot_class_distribution(target_column, figsize, title, xlabel, ylabel):
    """
        Example:
        plot_class_distribution(df['species'], (10, 5), 'Iris Species Distribution', 'Species', 'Count')
        It will plot a bar chart showing the distribution of the species in the iris dataset
    """
    
    value_counts = target_column.value_counts()
    plt.figure(figsize=figsize)
    value_counts.plot(kind='bar', color=plt.cm.Paired.colors[:len(value_counts)])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=0)
    plt.show()
    
    print(f"Class distribution:\n{value_counts}")
    
    
def boxplot_regression(df, col, target_col, figsize):
    """Plots a boxplot of a categorical column relative to a regression target variable."""
    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x=col, y=target_col, palette='Set2', hue=col)
    plt.title(f"Boxplot of '{target_col}' by '{col}'")
    plt.xlabel(col)
    plt.ylabel(target_col)
    plt.show()
    
def plot_categorical_distribution_with_target(df, col, target_col, figsize):
    """Plots the distribution of a categorical column relative to a target variable."""
    plt.figure(figsize=figsize)
    sns.countplot(data=df, x=col, hue=target_col, palette='Set2')
    plt.title(f"Distribution of '{col}' by '{target_col}'")
    plt.xlabel(target_col)
    plt.ylabel(col)
    plt.show()

def plot_numerical_with_target(df, col, target_col, figsize, type):
    """Plot a histogram of a numerical column colored by a target variable."""
    if type == "classification":
        plt.figure(figsize=figsize)
        sns.histplot(data=df, x=col, hue=target_col, bins=20, kde=True, palette='Set2')
        plt.title(f"Distribution of '{col}' by '{target_col}'")
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()
    else:
        plt.figure(figsize=figsize)
        sns.scatterplot(data=df, x=col, y=target_col)
        plt.title(f"Scatter plot of '{col}' vs '{target_col}'")
        plt.xlabel(col)
        plt.ylabel(target_col)
        plt.show()

def plot_distribution_and_trend(df, col, date_col, figsize=(), log_scale=True):
    """
    Plots distribution (with KDE + optional log scale) and yearly trend of a continuous target.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[col, date_col])
    df = df[df[col].notna()]

    # --- Distribution with KDE ---
    x = df[col].dropna()
    plt.figure(figsize=figsize if figsize else (6, 4))
    plt.hist(x, bins=50, edgecolor='black', alpha=0.6, density=True)
    kde = gaussian_kde(x)
    xs = np.linspace(x.min(), x.max(), 300)
    plt.plot(xs, kde(xs), color='red', lw=1.5, label='KDE')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # --- Optional log-scale plot ---
    if log_scale:
        plt.figure(figsize=figsize if figsize else (6, 4))
        plt.hist(np.log1p(x), bins=50, edgecolor='black', alpha=0.6, density=True)
        kde_log = gaussian_kde(np.log1p(x))
        xs_log = np.linspace(np.log1p(x).min(), np.log1p(x).max(), 300)
        plt.plot(xs_log, kde_log(xs_log), color='red', lw=1.5, label='KDE (log scale)')
        plt.title(f"Log-Scaled Distribution of {col}")
        plt.xlabel(f"log(1 + {col})")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    # --- Line chart per year ---
    df['year'] = df[date_col].dt.year
    yearly_mean = df.groupby('year')[col].mean()

    plt.figure(figsize=figsize if figsize else (6, 4))
    plt.plot(yearly_mean.index, yearly_mean.values, marker='o')
    plt.title(f"{col} Trend per Year")
    plt.xlabel("Year")
    plt.ylabel(f"Average {col}")
    plt.grid(alpha=0.3)
    plt.show()
