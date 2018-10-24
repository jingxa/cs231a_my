import numpy as np

if __name__ == "__main__":
    a = np.array([[1,3],[4,6],[7,9]])
    c = np.array([[4,5,],[6,7],[1,1]])
    b = np.array([4,5,6])

    d = np.r_[a,c]

    e = np.c_[a,c]

    print(d)

    print(e)