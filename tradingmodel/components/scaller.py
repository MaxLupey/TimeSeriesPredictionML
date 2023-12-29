from sklearn.preprocessing import MinMaxScaler
def scaller(X):
    scalers = {}
    for i in range(X.shape[1]):
        scalers[i] = MinMaxScaler()
        X[:, i, :] = scalers[i].fit_transform(X[:, i, :]) 

    print('min max scaller.........')
    return X