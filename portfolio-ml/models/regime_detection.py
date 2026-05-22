from sklearn.cluster import KMeans

def detect_regimes(features):

    model = KMeans(
        n_clusters=3,
        random_state=42
    )

    regimes = model.fit_predict(
        features
    )

    return regimes