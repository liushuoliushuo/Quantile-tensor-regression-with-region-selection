import numpy as np
import tensorly
import matplotlib.pyplot as plt

def genData_2D(n_train = 500, n_test = 200, region="one_rectangle", distribution = 'normal', p=20):

    weight = np.zeros((p, p), dtype=float)

    if region == "one_rectangle":

        length = p // 10
        weight[p // 2 - length: p//2 + length, p // 2 - length: p//2 + length]=1

    elif region == "two_rectangle":
        length = p // 5
        weight[p//2 - length: p//2, p//2 - length: p//2]=1
        weight[p//2:p//2+length, p//2:p//2+length]=1

    elif region == "three_rectangle":

        length = p // 5
        weight[p // 2 - length//2: p//2 + length//2, p // 2 - length//2: p//2 + length//2]=1
        weight[p // 2 - length//2 - length: p // 2 - length//2, p // 2 - length//2 - length: p // 2 - length//2]=1
        weight[p//2 + length//2:p//2 + length//2+length, p//2 + length//2:p//2 + length//2+length]=1

    else:
        print("Error in the selection of region")

    weight = tensorly.tensor(weight)

    x_train = np.random.randn(n_train, p, p)
    x_test = np.random.randn(n_test, p, p)

    x_train = tensorly.tensor(x_train)
    x_test = tensorly.tensor(x_test)

    y_train = tensorly.dot(tensorly.base.partial_tensor_to_vec(x_train, skip_begin=1),
                           tensorly.base.tensor_to_vec(weight))
    y_test = tensorly.dot(tensorly.base.partial_tensor_to_vec(x_test, skip_begin=1),
                          tensorly.base.tensor_to_vec(weight))

    if distribution == 'normal':
        y_train = y_train + np.random.normal(loc=0, scale=1, size=n_train)
        y_test = y_test + np.random.normal(loc=0, scale=1, size=n_test)
    elif distribution == 'cauchy':
        y_train = y_train + np.random.standard_cauchy(n_train)
        y_test = y_test +np.random.standard_cauchy(n_test)
    elif distribution == 't':
        y_train = y_train + np.random.standard_t(2, n_train)
        y_test = y_test + np.random.standard_t(2, n_test)
    elif distribution == 'laplace':
        y_train = y_train +np.random.laplace(0, 3, n_train)
        y_test = y_test +np.random.laplace(0, 3, n_test)
    elif distribution == 'outlier':
        outlier = np.hstack([np.ones(int(n_train * 0.1)) * 10, np.ones(int(n_train * 0.1)) * -10,
                             np.zeros(n_train - int(n_train * 0.2))])
        y_train = y_train + np.random.normal(loc=0, scale=1, size=n_train) + outlier
        y_test = y_test + np.random.normal(loc=0, scale=1, size=n_test)

    y_train = tensorly.tensor(y_train)
    y_test = tensorly.tensor(y_test)

    return {'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train,
            'y_test': y_test,
            'coef': weight}


def genData_3D(n_train = 500, n_test = 200, region="one_brick", distribution = 'normal', p=10):

    weight = np.zeros((p, p, p), dtype=float)

    if region == 'one_brick':
        length = p // 3
        core = np.ones((length, length, length))
        corner = [p//3+1, p//3+1, p//3+1]
        weight[corner[0]: corner[0] + length, corner[1]: corner[1] + length, corner[2]: corner[2] + length] = core

    elif region == 'two_brick':

        length = p//3
        core = np.ones((length, length, length))
        corner = [p//5, p//2]

        weight[corner[0]:corner[0]+length, corner[0]:corner[0]+length, corner[0]:corner[0]+length] = core
        weight[corner[1]:corner[1] + length, corner[1]:corner[1] + length, corner[1]:corner[1] + length] = core

    elif region == 'three_brick':

        length = p//3
        core = np.ones((length, length, length))
        corner = [1,4,7]

        weight[corner[0]:corner[0]+length, corner[0]:corner[0]+length, corner[0]:corner[0]+length] = core
        weight[corner[1]:corner[1] + length, corner[1]:corner[1] + length, corner[1]:corner[1] + length] = core
        weight[corner[2]:corner[2] + length, corner[2]:corner[2] + length, corner[2]:corner[2] + length] = core

    else:
        print("Error in the selection of region")

    weight = tensorly.tensor(weight)

    x_train = np.random.uniform(size=(n_train, p, p, p))
    x_test = np.random.uniform(size=(n_test, p, p, p))

    x_train = tensorly.tensor(x_train)
    x_test = tensorly.tensor(x_test)

    y_train = tensorly.dot(tensorly.base.partial_tensor_to_vec(x_train, skip_begin=1),
                           tensorly.base.tensor_to_vec(weight))
    y_test = tensorly.dot(tensorly.base.partial_tensor_to_vec(x_test, skip_begin=1),
                          tensorly.base.tensor_to_vec(weight))

    if distribution == 'normal':
        y_trian = y_train + np.random.normal(loc=0, scale=1, size=n_train)
        y_test = y_test + np.random.normal(loc=0, scale=1, size=n_test)
    elif distribution == 'cauchy':
        y_train = y_train + np.random.standard_cauchy(n_train)
        y_test = y_test + np.random.standard_cauchy(n_test)
    elif distribution == 't':
        y_train = y_train + np.random.standard_t(2, n_train)
        y_test = y_test + np.random.standard_t(2, n_test)
    elif distribution == 'laplace':
        y_train = y_train + np.random.laplace(0, 3, n_train)
        y_test = y_test + np.random.laplace(0, 3, n_test)
    elif distribution == 'outlier':
        outlier = np.hstack([np.ones(int(n_train * 0.1)) * 10, np.ones(int(n_train * 0.1)) * -10,
                             np.zeros(n_train - int(n_train * 0.2))])
        y_train = y_train + np.random.normal(loc=0, scale=1, size=n_train) + outlier
        y_test = y_test + np.random.normal(loc=0, scale=1, size=n_test)

    y_train = tensorly.tensor(y_train)
    y_test = tensorly.tensor(y_test)

    return {'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train,
            'y_test': y_test,
            'coef': weight}

def scatter_3d(data, filepath=None):
    plt.figure()
    mycolormap = plt.get_cmap('Greens')

    data = abs(data) / abs(data).max()
    x = []
    y = []
    z = []
    c = []
    for i in range(0, data.shape[0]):
        for j in range(0, data.shape[1]):
            for k in range(0, data.shape[2]):
                tempc = mycolormap(data[i][j][k])
                x.append(i)
                y.append(j)
                z.append(k)
                if data[i][j][k] != 0:
                    c.append((tempc[0], tempc[1], tempc[2], 0.5))
                else:
                    c.append((tempc[0], tempc[1], tempc[2], 0))

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(x,y,z, c=c, s = 80)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_box_aspect([1, 1, 1])
    if filepath == None:
        plt.show()
    else:
        plt.savefig(filepath, bbox_inches='tight',dpi=350)
    plt.close()


