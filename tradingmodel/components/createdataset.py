import numpy as np
# def createdataset(dataar, date, close, high, mode):
#     interval = 30
#     Y = []
#     X = []
#     date2 = []
#     close2 = []
#     high2 = []
#     for e in range(dataar.shape[0]-interval-1):
#     # if dataar[e+interval+1][1] > dataar[e+interval][3]*1.001:
#     #     a = 1
#     # elif dataar[e+interval+1][2]< dataar[e+interval][3]*0.999:
#     #     a = 0
#         if mode=='buy':
#             if dataar[e+interval+1][1] > dataar[e+interval][3]*1.001 and not dataar[e+interval+1][2] > dataar[e+interval][3]*0.999:
#                 a = 1
#             else :
#                 a = 0
#         elif mode=='sell':
#             if not dataar[e+interval+1][1] > dataar[e+interval][3]*1.15 and dataar[e+interval+1][2] > dataar[e+interval][3]*0.999:
#                 a = 1
#             else :
#                 a = 0
#         Y.append(a)
#         date2.append(date[e])
#         close2.append(close[e])
#         high2.append(high[e])
#         XX=[]
#         for a in range(30):
#             XX.append(dataar[e+a])
#         X.append(XX)
#     # X = dataar[:dataar.shape[0]-interval].astype(np.float32)
#     X = np.array(X).astype(np.float32)
#     Y = np.array(Y)
#     dates = np.array(date2)
#     closes = np.array(close2)
#     highes = np.array(high2)
#     # Y = np.expand_dims(Y, axis=0)
#     print('00000 :',len([a  for a in Y if a==0]),',  1111 :', len([a for a in Y if a==1]) )
#     print('dataset is created........')
#     return X, Y, dates, closes, highes
def createdataset(dataar, mode):
    interval = 30
    Y = []
    X = []
    for e in range(dataar.shape[0]-interval-1):
    # if dataar[e+interval+1][1] > dataar[e+interval][3]*1.001:
    #     a = 1
    # elif dataar[e+interval+1][2]< dataar[e+interval][3]*0.999:
    #     a = 0
        if mode=='buy':
            if dataar[e+interval+1][1] > dataar[e+interval][3]*1.001 and not dataar[e+interval+1][2] > dataar[e+interval][3]*0.999:
                a = 1
            else :
                a = 0
        elif mode=='sell':
            if not dataar[e+interval+1][1] > dataar[e+interval][3]*1.15 and dataar[e+interval+1][2] > dataar[e+interval][3]*0.999:
                a = 1
            else :
                a = 0
        Y.append(a)
        XX=[]
        for a in range(30):
            XX.append(dataar[e+a])
        X.append(XX)
    # X = dataar[:dataar.shape[0]-interval].astype(np.float32)
    X = np.array(X).astype(np.float32)
    Y = np.array(Y)
    # Y = np.expand_dims(Y, axis=0)
    print('00000 :',len([a  for a in Y if a==0]),',  1111 :', len([a for a in Y if a==1]) )
    print('dataset is created........')
    return X, Y