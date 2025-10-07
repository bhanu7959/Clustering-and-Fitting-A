"""
Wine Clustering and Fitting Analysis
This script performs statistical analysis, clustering, and linear regression
on wine chemical composition data.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def plot_relational_plot(df):
    """Create a scatter plot of the first two numeric columns."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1], ax=ax)
    ax.set_title(f'{df.columns[0]} vs {df.columns[1]}')
    plt.savefig('relational_plot.png')
    plt.show()
    plt.close()
    return


def plot_categorical_plot(df):
    """Create a box plot of the first two numeric columns."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x=df.columns[0], y=df.columns[1], ax=ax)
    ax.set_title(f'Distribution of {df.columns[1]} by {df.columns[0]}')
    plt.savefig('categorical_plot.png')
    plt.show()
    plt.close()
    return


def plot_statistical_plot(df):
    """Create a histogram with KDE of the first column."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=df, x=df.columns[0], kde=True, ax=ax)
    ax.set_title(f'Distribution of {df.columns[0]}')
    plt.savefig('statistical_plot.png')
    plt.show()
    plt.close()
    return


def statistical_analysis(df, col: str):
    """Calculate statistical moments for a given column."""
    data = df[col].dropna()
    mean = np.mean(data)
    stddev = np.std(data)
    skew = ss.skew(data)
    excess_kurtosis = ss.kurtosis(data)
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """Preprocess the wine dataset."""
    # Basic data exploration
    print("\nData Summary:")
    print(df.describe())
    # First 5 rows
    print("\nFirst 5 rows:")
    print(df.head())
    # Correlation Matrix
    print("\nCorrelation Matrix:")
    print(df.corr())
    # Handle missing values
    df = df.dropna()
    # Remove outliers using z-score (threshold = 3)
    numeric_cols = df.select_dtypes(include=['number']).columns
    z_scores = np.abs(ss.zscore(df[numeric_cols]))
    df = df[(z_scores < 3).all(axis=1)]
    # Return
    return df


def writing(moments, col):
    """Print statistical analysis results with interpretation."""
    print(f'\nFor the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    # Interpret skewness
    skew = moments[2]
    if skew > 1:
        skew_text = "right skewed"
    elif skew < -1:
        skew_text = "left skewed"
    else:
        skew_text = "not significantly skewed"
    # Interpret kurtosis
    kurt = moments[3]
    if kurt > 1:
        kurt_text = "leptokurtic (heavy-tailed)"
    elif kurt < -1:
        kurt_text = "platykurtic (light-tailed)"
    else:
        kurt_text = "mesokurtic (normal-tailed)"
    # Print
    print(f'The data is {skew_text} and {kurt_text}.')
    return


def perform_clustering(df, col1, col2):
    """Perform K-means clustering on selected columns."""
    def plot_elbow_method(data):
        """Plot elbow method for determining optimal clusters."""
        inertias = []
        for k in range(1, 8):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(range(1, 8), inertias, marker='o')
        ax.set_title('Elbow Method for Optimal k')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('Inertia')
        plt.savefig('elbow_plot.png')
        plt.show()
        plt.close()
        return

    def one_silhouette_inertia(data, n_clusters=3):
        """Calculate silhouette score and inertia."""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data)
        _score = silhouette_score(data, labels)
        _inertia = kmeans.inertia_
        return _score, _inertia, kmeans
    # Prepare data
    data = df[[col1, col2]].dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    # Determine optimal clusters
    plot_elbow_method(scaled_data)
    # Perform clustering with 3 clusters
    score, inertia, kmeans = one_silhouette_inertia(scaled_data, n_clusters=3)
    labels = kmeans.labels_
    # Get cluster centers in original scale
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    xkmeans = centers[:, 0]
    ykmeans = centers[:, 1]
    cenlabels = [f'Cluster {i+1}' for i in range(3)]
    # print
    print(f"\nClustering Metrics for {col1} vs {col2}:")
    print(f"Silhouette Score: {score:.3f} (Higher is better)")
    print(f"Inertia: {inertia:.2f} (Lower is better)")
    # Return
    return labels, data, xkmeans, ykmeans, cenlabels


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    """Plot clustered data with centers."""
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(*data.iloc[:, :2].values.T, c=labels, cmap='viridis')
    ax.scatter(xkmeans, ykmeans, c='red', marker='X', s=200, label='Centers')
    for i, label in enumerate(centre_labels):
        ax.annotate(
           label, (xkmeans[i], ykmeans[i]),
           textcoords="offset points",
           xytext=(0, 5),
           ha='center'
        )
    ax.set_title(f'K-means Clustering: {data.columns[0]} vs {data.columns[1]}')
    ax.set_xlabel(data.columns[0])
    ax.set_ylabel(data.columns[1])
    ax.legend(*scatter.legend_elements(), title="Clusters")
    plt.savefig('clustering.png', dpi=300)
    plt.show()
    plt.close()
    return


def perform_fitting(df, col1, col2):
    """Perform linear regression on selected columns."""
    # Prepare data
    data = df[[col1, col2]].dropna()
    X = data[col1].values.reshape(-1, 1)
    y = data[col2].values
    # Fit model
    model = LinearRegression()
    model.fit(X, y)
    # Predict across range
    x = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x)
    # Print
    print(f"\nRegression Results for {col1} predicting {col2}:")
    print(f"Slope: {model.coef_[0]:.4f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"R-squared: {model.score(X, y):.4f}")
    # Return
    return data, x, y_pred


def plot_fitted_data(data, x, y):
    """Plot data with regression line."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(data.iloc[:, 0], data.iloc[:, 1], alpha=0.6, label='Data')
    ax.plot(x, y, color='red', linewidth=2, label='Regression Line')
    ax.set_title(f'Linear Regression: {data.columns[0]} vs {data.columns[1]}')
    ax.set_xlabel(data.columns[0])
    ax.set_ylabel(data.columns[1])
    ax.legend()
    plt.savefig('fitting.png', dpi=300)
    plt.show()
    plt.close()
    return


def main():
    """Main analysis pipeline."""
    # Load data
    try:
        df = pd.read_csv('data.csv')
        print("Data loaded successfully with shape:", df.shape)
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    # Preprocess
    df = preprocessing(df)
    # columns for analysis
    stat_col = 'Alcohol'  # For statistical analysis
    cluster_cols = ('Alcohol', 'Malic_Acid')  # For clustering
    fit_cols = ('Alcohol', 'Color_Intensity')  # For fitting
    # Generate plots
    plot_relational_plot(df[list(cluster_cols)])
    plot_statistical_plot(df[[stat_col]])
    plot_categorical_plot(df[list(cluster_cols)])
    # Statistical analysis
    moments = statistical_analysis(df, stat_col)
    writing(moments, stat_col)
    # Clustering
    clustering_results = perform_clustering(df, *cluster_cols)
    plot_clustered_data(*clustering_results)
    # Fitting
    fitting_results = perform_fitting(df, *fit_cols)
    plot_fitted_data(*fitting_results)
    # Print
    print("\nAnalysis complete. Check generated plots.")
    return


if __name__ == '__main__':
    main()
