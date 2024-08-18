import pandas as pd
from sklearn.cluster import KMeans # This line imports the KMeans class from the scikit-learn library, which is used for clustering data.
import matplotlib.pyplot as plt

# Store the cereal data from cereal.csv into a python dataframe.
df = pd.read_csv("cereal.csv")

# Extracting the relevant columns for clustering
X = df[['cups', 'sugars']]

# Plotting the initial scatter plot using 'cups' and 'sugars'
plt.scatter(X['cups'], X['sugars'])
plt.xlabel("Cups", fontsize=18)
plt.ylabel("Sugars", fontsize=18)
plt.title("Initial Scatter Plot")
plt.savefig("plot.png")
plt.show()

# Determine the number of clusters (k) based on the visual inspection of the initial scatter plot
k = 3

# Create a KMeans object named km with the chosen number of clusters and set random state for reproducibility
km = KMeans(n_clusters=k, random_state=0)

# Fit the KMeans model to the data X, clustering the data points into the specified number of clusters
km.fit(X)

# Predict the cluster labels for the data points in X using the trained KMeans model and store them in new_labels
new_labels = km.labels_

# Plot the clustered data with the appropriate color map
plt.scatter(X['cups'], X['sugars'], c=new_labels, cmap='gist_rainbow')
plt.xlabel('Cups', fontsize=18)
plt.ylabel('Sugars', fontsize=18)
plt.title("Clustered Data")
plt.savefig("Prediction.png")
plt.show()
