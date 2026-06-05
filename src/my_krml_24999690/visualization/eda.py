import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_distribution(df, col, hue=None, title=None, figsize=(8, 5)):
    """Plots the distribution of a numerical column (Histogram + KDE)."""
    plt.figure(figsize=figsize)
    sns.histplot(data=df, x=col, hue=hue, kde=True, palette='Set2' if hue else None)
    plt.title(title if title else f"Distribution of {col}" + (f" by {hue}" if hue else ""))
    plt.tight_layout()
    plt.show()

def plot_barchart(df, col, hue=None, title=None, figsize=(8, 5)):
    """Plots a beautiful bar chart (frequency count) of a categorical column or target class."""
    plt.figure(figsize=figsize)
    sns.countplot(data=df, x=col, hue=hue, palette='Set2')
    plt.title(title if title else f"Count of {col}" + (f" by {hue}" if hue else ""))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Alias for backward compatibility
plot_categorical = plot_barchart

def plot_boxplot(df, cat_col, num_col, hue=None, title=None, figsize=(8, 5)):
    """Plots a boxplot to compare a numerical column across categories."""
    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x=cat_col, y=num_col, hue=hue, palette='Set2')
    plt.title(title if title else f"Boxplot of {num_col} by {cat_col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_scatter(df, x_col, y_col, hue=None, trendline=False, title=None, figsize=(8, 5)):
    """Plots a scatter plot between two numerical columns, optionally with a trendline."""
    plt.figure(figsize=figsize)
    if trendline and not hue:
        # regplot computes and draws a linear regression line
        sns.regplot(data=df, x=x_col, y=y_col, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, palette='Set2', alpha=0.7)
    
    plt.title(title if title else f"Scatter plot: {x_col} vs {y_col}")
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df, columns=None, title=None, figsize=(10, 8)):
    """Plots a beautiful correlation heatmap for numerical columns."""
    plt.figure(figsize=figsize)
    
    # Filter numerical columns or use provided list
    data = df[columns] if columns else df.select_dtypes(include=[np.number])
        
    corr = data.corr()
    
    # Create a mask for the upper triangle for a cleaner look
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, square=True)
    plt.title(title if title else "Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def plot_pairplot(df, columns=None, hue=None, title=None):
    """Plots pairwise relationships across multiple numerical columns."""
    if columns:
        data = df[columns + ([hue] if hue and hue not in columns else [])]
    else:
        # Auto-select numerical columns + hue
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if hue and hue not in num_cols:
            num_cols.append(hue)
        data = df[num_cols]
        
    # corner=True makes it a beautiful lower-triangle matrix
    sns.pairplot(data, hue=hue, palette='Set2', corner=True)
    plt.suptitle(title if title else "Pairwise Relationships", y=1.02)
    plt.show()
