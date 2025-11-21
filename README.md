# Educational Movie Clustering Project report

## Project Overview

This project delves into the fascinating world of movie data to discover natural groupings, or "clusters," among films. Imagine a streaming service or a film studio that wants to understand its vast catalog better, or even predict where a new movie might fit into the market. Movie clustering helps achieve this by categorizing films into distinct groups based on various characteristics, from genre and cast to plot summaries and financial performance.

The core goal is to build a robust system that can automatically assign a new movie to a specific cluster and provide a clear explanation of *why* it belongs there. This not only enhances content organization and recommendation systems but also offers valuable insights for strategic decision-making in the entertainment industry.

## Data Preparation

The journey began with two rich datasets: `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`, containing information on thousands of films. These were merged using `id` and `movie_id` to create a comprehensive dataset.

**Key Data Cleaning and Preprocessing Steps:**

*   **Handling Duplicates and Nulls:** Redundant columns like `movie_id` and `title_y` were dropped. Rows with missing `overview`, `release_date`, or `runtime` (which were very few) were removed. The `homepage` column, containing numerous nulls and providing little clustering value, was also dropped. The `tagline` column's nulls were thoughtfully filled with empty strings, preserving the data and allowing the clustering algorithm to distinguish between movies with and without taglines.
*   **Feature Transformation:** Stringified JSON columns (`genres`, `keywords`, `cast`, `crew`, etc.) were safely parsed using `ast.literal_eval` to extract meaningful information like genre names, top cast members, and the director.
*   **Numerical Feature Engineering:** New features were created, such as `profit_ratio`, calculated from `revenue` and `budget`. Highly skewed numerical features like `budget`, `revenue`, and `popularity` were log-transformed (`np.log1p`) to normalize their distributions, which is crucial for distance-based algorithms. The `release_date` was converted to `release_year` for easier numerical analysis.
*   **Scaling Numerical Features:** All numerical and engineered features (`vote_average`, `vote_count`, `popularity`, `budget_log`, `revenue_log`, `runtime`, `profit_ratio`, `release_year`) were standardized using `StandardScaler` to ensure that no single feature dominated the clustering process due to its scale.
*   **Exploratory Data Analysis (EDA):** Density plots, box plots, and correlation heatmaps were generated using `Matplotlib` and `Seaborn` to understand feature distributions, identify outliers, and reveal relationships between numerical variables. This confirmed the necessity of log transformation and robust scaling for many features.

## Advanced Feature Engineering

Transforming raw movie data into meaningful numerical features is a critical step. This project employed advanced natural language processing (NLP) techniques to capture the essence of textual information:

*   **Doc2Vec Embeddings for Descriptive Tags:** For categorical and descriptive text fields like `genres`, `keywords`, `cast`, and `director`, the combined cleaned lists were converted into a single string (`all_features_str`). `NLTK` was used for tokenization, and `Gensim`'s `Doc2Vec` model was trained on these aggregated features. This process generated 100-dimensional vector embeddings for each movie, effectively capturing the semantic relationships between these descriptive tags.
*   **Sentence-BERT Embeddings for Plot Summaries:** The `overview` (plot summary) field, being a longer, more descriptive text, was processed using `SentenceTransformer` with the 'all-MiniLM-L6-v2' model. This yielded rich 384-dimensional vector embeddings that encapsulate the semantic meaning of each movie's plot, going beyond simple keyword matching.
*   **Feature Consolidation:** The numerical, Doc2Vec, and Sentence-BERT embeddings were concatenated to form a comprehensive feature matrix, ready for dimensionality reduction.

## Dimensionality Reduction with UMAP

High-dimensional data can pose challenges for clustering algorithms. `UMAP` (Uniform Manifold Approximation and Projection) was employed to reduce the complexity of the feature space while preserving the underlying structure and relationships within the data.

*   **UMAP for Visualization (2D):** A UMAP model with `n_components=2` was applied to the full feature matrix. This 2-dimensional representation (`umap_1`, `umap_2`) is invaluable for visualizing the clusters and understanding their separation in a human-interpretable space.
*   **UMAP for Clustering (50D):** A separate UMAP model with `n_components=50` was used to create a higher-dimensional, yet still reduced, feature space specifically optimized for clustering. This effectively cleanses the data of noise and minor variations, allowing clustering algorithms to find more robust and meaningful patterns. Cosine distance was chosen as the metric for UMAP, which is suitable for high-dimensional embeddings like those generated by Doc2Vec and Sentence-BERT.

All UMAP transformations were performed using the `umap-learn` Python library, known for its efficiency and effectiveness in manifold learning.

## Clustering Models and Evaluation

To identify the best way to group movies, three prominent clustering algorithms were explored: `K-Means`, `HDBSCAN`, and `Gaussian Mixture Model (GMM)`.

*   **K-Means:**
    *   **Mechanism:** An iterative algorithm that partitions data points into a predefined number of clusters (K), aiming to minimize the variance within each cluster.
    *   **K Selection:** The Elbow Method was used to determine the optimal number of clusters for K-Means, suggesting K=5 as a reasonable choice.
    *   **Implementation:** `sklearn.cluster.KMeans`
*   **HDBSCAN:**
    *   **Mechanism:** A density-based algorithm that identifies clusters of varying shapes and densities and can detect outliers as noise. It does not require specifying the number of clusters beforehand.
    *   **Implementation:** `hdbscan` library
*   **Gaussian Mixture Model (GMM):**
    *   **Mechanism:** A probabilistic model that assumes data points are generated from a mixture of several Gaussian distributions. It assigns each data point a probability of belonging to each cluster.
    *   **Implementation:** `sklearn.mixture.GaussianMixture`

**Evaluation Metrics:**

To quantitatively compare the models, three key metrics were used:

*   **Silhouette Score:** Measures how similar an object is to its own cluster compared to other clusters. Higher values indicate better-defined clusters.
*   **Davies-Bouldin Index:** Measures the average similarity ratio between each cluster and its most similar cluster. Lower values indicate better separation.
*   **Calinski-Harabasz Index:** Measures the ratio of between-cluster dispersion to within-cluster dispersion. Higher values indicate better-defined clusters.

While HDBSCAN showed superior Silhouette and Davies-Bouldin scores, and K-Means had the highest Calinski-Harabasz score, **K-Means was ultimately selected as the best-performing model.** This decision was based not only on quantitative metrics but also on a qualitative assessment of its cluster interpretability and its ability to classify new movies into intuitively understandable categories. HDBSCAN, while statistically strong, often produced less intuitive classifications for some test cases.

The clusters generated by K-Means were then visualized using the 2D UMAP projection, clearly showing distinct groupings of movies.

## Cluster Interpretation

Each of the 5 K-Means clusters was meticulously analyzed and profiled to give it a descriptive name. This involved examining the mean Z-scores of numerical features (like `vote_average`, `profit_ratio`, `release_year`, `popularity`) and the most frequent categorical features (`genres`, `keywords`, `cast`, `director`) within each cluster.

## Practical Application: `predict_and_explain_cluster` Function

To demonstrate the real-world utility of this project, a `predict_and_explain_cluster` function  that does the same steps to prepare the data like what have done in the whole project and predicts the most suitable it cluster it can get for the input data was developed. This function allows users to input details for a new, unseen movie and receive:
1.  **Predicted Cluster ID and Name:** .
2.  **Second Closest Cluster ID and Name**
3.  **Analytical Explanation:**

## Conclusion and Impact

This educational movie clustering project shows advenced analysis techniques and advanced techniques that converted the strind data into numerical data using Doc2Vec and Sentence_Bert which are NLP techniques and reducing the dimensions of the data using UMAP technique so the data can be understandble for the clustering model and this project showed significantly good clustering results and with more availble data this project idea would be more better in the real word business

## Trying the models 
For trying the models without running the whole project file you should download the MovieClusterFunction file and to use the function dowload with it all the joblib files here in this repository 

## Used data
Used data is from kaggle suddenly I coudn't puplish the file because github didn't accept it due to its size but there is the link on kaggle: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata
