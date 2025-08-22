import QTR, data
import numpy as np
import matplotlib.pyplot as plt

def calculate_rmse(array1, array2):
    mse = np.mean((array1 - array2) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def calculate_mae(array1, array2):
    mae = np.mean(np.abs(array1 - array2))
    return mae

print("======================2D simulation======================")

# generate simulation data
sample = data.genData_2D(n_train=500,
                         n_test=200,
                         region='one_rectangle',
                         distribution='cauchy',
                         p=20)

# model
model = QTR.QTR(tau=0.5,
                rank=1,
                model_lambda=5,
                penalty='scad')

# fitting the model
model.fit(sample['y_train'], sample['x_train'])

# BIC
print("BIC: ", model.BIC())

# estimation performance
print("RMSE: ", calculate_rmse(model.W, sample['coef']))

# prediction performance
y_pred = model.predict(sample['x_test'])
print("MAPE: ", calculate_mae(y_pred.reshape(-1,), sample['y_test']))

# region selection performance
est_coef = model.W.flatten()
actual_coef = sample['coef'].flatten()
TP = np.sum((actual_coef != 0) & (est_coef != 0))
FP = np.sum((actual_coef == 0) & (est_coef != 0))
TN = np.sum((actual_coef == 0) & (est_coef == 0))
FN = np.sum((actual_coef != 0) & (est_coef == 0))
print("TP: ", TP)
print("FP: ", FP)
print("TN: ", TN)
print("FN: ", FN)

# true and estimated coefficient tensor
plt.subplot(1,2,1)
plt.imshow(sample['coef'], cmap="Reds")
plt.axis('off')
plt.title("true coefficient tensor")

plt.subplot(1,2,2)
plt.imshow(model.W, cmap="Reds")
plt.axis('off')
plt.title("estimated coefficient tensor")
plt.show()

print("======================3D simulation======================")

# generate simulation data
sample = data.genData_3D(n_train=800,
                         n_test=200,
                         region='one_brick',
                         distribution='cauchy',
                         p=10)

# model
model = QTR.QTR(tau=0.5,
                rank=1,
                model_lambda=5,
                penalty='scad')

# fitting the model
model.fit(sample['y_train'], sample['x_train'])

# BIC
print("BIC: ", model.BIC())

# estimation performance
print("RMSE: ", calculate_rmse(model.W, sample['coef']))

# prediction performance
y_pred = model.predict(sample['x_test'])
print("MAPE: ", calculate_mae(y_pred.reshape(-1,), sample['y_test']))

# region selection performance
est_coef = model.W.flatten()
actual_coef = sample['coef'].flatten()
TP = np.sum((actual_coef != 0) & (est_coef != 0))
FP = np.sum((actual_coef == 0) & (est_coef != 0))
TN = np.sum((actual_coef == 0) & (est_coef == 0))
FN = np.sum((actual_coef != 0) & (est_coef == 0))
print("TP: ", TP)
print("FP: ", FP)
print("TN: ", TN)
print("FN: ", FN)

# true and estimated coefficient tensor
data.scatter_3d(sample['coef'])
data.scatter_3d(model.W)

