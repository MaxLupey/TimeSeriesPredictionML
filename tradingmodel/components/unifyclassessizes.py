import numpy as np
from sklearn.utils import shuffle
def unify_classes_sizes(X , Y):
    y1, x1, d1,c1, y2,x2, d2,c2 = [],[],[],[],[],[],[],[]
    for e,a in enumerate(Y):
        if a == 1:
            y1.append(a)
            x1.append(X[e])
        if a == 0:
            y2.append(a)
            x2.append(X[e])
        
    a = [len(y1), len(y2)]
    n = a[np.argmin(a)]
    
    y1, x1 = shuffle(y1, x1, random_state=42)
    y2, x2 = shuffle(y2, x2, random_state=42)
    y1, x1, y2, x2, = y1[:n], x1[:n], y2[:n], x2[:n]

    X = x1+x2
    Y = y1+y2
    
    X,y = shuffle(np.array(X), np.array(Y), random_state=42)
    print('unnify classes........')
    return X,y