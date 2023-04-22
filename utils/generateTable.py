# Generating the lookup table
def generateTable(data, k=4):
    T = {}
    for i in range(len(data) - k):
        X = data[i:i + k]
        Y = data[i + k]
        # print("X  %s and Y %s  "%(X,Y))

        if T.get(X) is None:
            T[X] = {}
            T[X][Y] = 1
        else:
            if T[X].get(Y) is None:
                T[X][Y] = 1
            else:
                T[X][Y] += 1

    return T


