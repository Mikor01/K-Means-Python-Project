import numpy as np

class MyKMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _sq_distances(self, X, C):
        diff = X[:, None, :] - C[None, :, :]

        return np.sum(diff**2, axis=2)

    def fit(self, X):
        # Check value constrains
        X = np.asarray(X, dtype = float)

        if X.ndim != 2:
            raise ValueError("X has to be a 2D table (n_samples, n_attributes)")

        n, d = X.shape

        if self.n_clusters < 1 or self.n_clusters > n:
            raise ValueError("n_clusters value must be between 1 and n_samples")

        # Select n_clusters random centroids from X
        rng = np.random.default_rng(self.random_state)

        idx = rng.choice(n, size=self.n_clusters, replace=False)

        centroids = X[idx].copy()

        for it in range(1, self.max_iter+1):
            dist = self._sq_distances(X,centroids)

            labels = np.argmin(dist, axis=1)

            new_centroids = centroids.copy()
            for j in range(self.n_clusters):
                points_j = X[labels == j]
                if len(points_j) > 0:
                    new_centroids[j] = points_j.mean(axis=0)

            shift = np.sum((new_centroids - centroids) ** 2)
            centroids = new_centroids

            if(shift <= self.tol):
                break

        # recalculate for the latest centroids from the last loop
        dist = self._sq_distances(X, centroids)
        labels = np.argmin(dist, axis=1)

        self.cluster_centers_ = centroids
        self.labels_ = labels
        self.n_iter_ = it
        self.inertia_ = np.sum(dist[np.arange(n), labels])

        return self

    def predict(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise ValueError("Call fit() before predict()")

        X = np.asarray(X, dtype=float)
        dist = self._sq_distances(X,self.cluster_centers_)  # distances to the final centroids
        return np.argmin(dist, axis=1)                      # index of the nearest centroid per row

    def fit_predict(self, X):
        return self.fit(X).predict(X)
