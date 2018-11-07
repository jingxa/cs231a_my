import  numpy as np



if __name__ == "__main__":


    a = np.array([1,2,3,4,6,9,7,5])
    b = a.reshape((8,1)).flatten()

    print(a)
    print(b)

    idx = np.argsort(a)
    idx2 = np.argsort(-b)

    print(idx, idx.shape)
    print(idx2, idx2.shape)
    print(5/2)