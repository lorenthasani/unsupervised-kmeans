# Customer Segmentation using K-Means Clustering

A machine learning project that implements K-Means clustering to segment customers based on their annual income and spending score. This unsupervised learning approach helps businesses understand customer behavior patterns and develop targeted marketing strategies.

## Project Overview

This project demonstrates the application of K-Means clustering algorithm for customer segmentation. By analyzing customer data including annual income and spending scores, the model identifies distinct customer groups, enabling businesses to:

- Understand customer behavior patterns
- Develop targeted marketing campaigns
- Optimize product offerings for different customer segments
- Make data-driven business decisions

## Dataset

The project uses the Mall Customers Dataset (`Mall_Customers.csv`) which contains the following features:

- **CustomerID**: Unique identifier for each customer
- **Gender**: Customer's gender
- **Age**: Customer's age
- **Annual Income (k$)**: Customer's annual income in thousands of dollars
- **Spending Score (1-100)**: Score assigned based on customer behavior and spending nature

## Technologies Used

- Python 3.x
- Jupyter Notebook
- Pandas - Data manipulation and analysis
- NumPy - Numerical computing
- Matplotlib - Data visualization
- Seaborn - Statistical data visualization
- Scikit-learn - Machine learning library for K-Means implementation

## Project Structure

```
unsupervised-kmeans/
│
├── Mall_Customers.csv           # Customer dataset
├── k_means_clustering.ipynb     # Main Jupyter notebook with implementation
├── .ipynb_checkpoints/          # Jupyter notebook checkpoints
└── README.md                    # Project documentation
```

## Getting Started

### Prerequisites

Make sure you have Python 3.x installed on your system. You can download it from [python.org](https://www.python.org/).

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/lorenthasani/unsupervised-kmeans.git
   cd unsupervised-kmeans
   ```

2. Install required packages
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. Launch Jupyter Notebook
   ```bash
   jupyter notebook
   ```

4. Open the notebook
   - Navigate to `k_means_clustering.ipynb` in the Jupyter interface
   - Run all cells to execute the analysis

## Methodology

The project follows these key steps:

1. Data Loading and Exploration
   - Import the Mall Customers dataset
   - Perform exploratory data analysis (EDA)
   - Visualize data distributions and relationships

2. Data Preprocessing
   - Handle missing values (if any)
   - Select relevant features for clustering
   - Scale/normalize data if necessary

3. Optimal Cluster Selection
   - Use the Elbow Method to determine optimal number of clusters
   - Analyze Within-Cluster Sum of Squares (WCSS)

4. K-Means Clustering
   - Apply K-Means algorithm with optimal cluster number
   - Fit the model to the data
   - Predict cluster assignments

5. Results Visualization
   - Create scatter plots showing customer segments
   - Visualize cluster centroids
   - Analyze cluster characteristics

6. Business Insights
   - Interpret clustering results
   - Provide actionable business recommendations

## Expected Results

The K-Means clustering typically identifies customer segments such as:

- High Income, High Spending: Premium customers
- High Income, Low Spending: Potential target for upselling
- Low Income, High Spending: Budget-conscious but active buyers
- Low Income, Low Spending: Price-sensitive customers
- Medium Income, Medium Spending: Average customers

## Key Insights

- Customer segmentation enables targeted marketing strategies
- Different clusters require different business approaches
- The model helps identify the most valuable customer segments
- Insights can drive inventory management and product development decisions

## Usage Example

```python
# Load the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Select features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X)

# Visualize results
plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=clusters, cmap='viridis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation using K-Means Clustering')
plt.show()
```

## Contributing

Contributions are welcome! If you'd like to contribute to this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Author

Lorent Hasani
- GitHub: [@lorenthasani](https://github.com/lorenthasani)

## Contact

If you have any questions or suggestions, feel free to reach out:

- Create an issue in this repository
- Connect with me on GitHub

## Acknowledgments

- Dataset source: Mall Customers Dataset
- Inspiration from various customer segmentation case studies
- Thanks to the open-source community for the amazing tools and libraries
