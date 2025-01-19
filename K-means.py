import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# K-means algorithm
def Kmeans(x, k):
    # Step 1: Initialize the centroids randomly
    counter = 0
    centroids = x[np.random.choice(range(len(x)), k, replace=False)]
    
    fig, ax = plt.subplots()
    scat = ax.scatter(x[:, 0], x[:, 1], c='gray', cmap='viridis')
    
    def update(frame):
        nonlocal centroids, counter
        # Step 2: Assign each data point to the nearest centroid
        clusters = np.argmin(np.linalg.norm(x[:, np.newaxis] - centroids, axis=2), axis=1)
        # Step 3: Update the centroids
        new_centroids = np.array([x[clusters == i].mean(axis=0) for i in range(k)])
        counter += 1
        if np.all(centroids == new_centroids):
            ani.event_source.stop()
        centroids = new_centroids
        scat.set_array(clusters)
        return scat,
    
    ani = FuncAnimation(fig, update, frames=range(100), interval=200, repeat=False)
    plt.show()
    
    return counter

# Generate Sample Data (2D) 
np.random.seed(0)
x = np.random.randn(1000, 2)
# Apply K-means algorithm
counter = Kmeans(x, 3)
print(counter)



