import numpy as np
import csv

def mse(y, y_pred):
    return np.mean((y - y_pred)**2)

def coefficientCalculation(X, y):
    # Calculating coefficients using the linear algebra equation:
    b_hat = np.linalg.inv(np.dot(X.T, X))
    b_hat = np.dot(b_hat, X.T)
    b_hat = np.dot(b_hat, y)
    return b_hat


with open("Football_players.csv") as f:
    csv_list = list(csv.reader(f))

age_set = np.array([])
hgt_set = np.array([])
mnt_set = np.array([])
skl_set = np.array([])
sal_set = np.array([])

# Extracting data into lists, creating X and y:
for row in csv_list[1:]:
    age_set = np.append(age_set, int(row[4]))
    hgt_set = np.append(hgt_set, int(row[5]))
    mnt_set = np.append(mnt_set, int(row[6]))
    skl_set = np.append(skl_set, int(row[7]))
    sal_set = np.append(sal_set, int(row[8]))

# Forming the input(X) and the output(y)
ones = np.ones((len(age_set)))
X = np.column_stack((ones, age_set, hgt_set, mnt_set, skl_set))
y = sal_set

X_train = X[:80]
y_train = y[:80]
X_test = X[80:]
y_test = y[80:]

# Calculating the coefficients & predictions:
coefficients = coefficientCalculation(X_train, y_train)
y_hat = np.dot(X_test, coefficients)

# Calculating and displaying MSE for this set of predictions:
print("Test error w/ 80-20 split:")
print("MSE:" , mse(y_test, y_hat), "\n")

# Calculating the coefficients & predictions:
coefficients = coefficientCalculation(X, y)
y_hat = np.dot(X, coefficients)

# Calculating and displaying MSE for this set of predictions:
print("Training error:")
print("MSE:", mse(y, y_hat), "\n")

print("-----------------DATA W/ RANDOM COLN------------")
print("")

rand_set = np.random.randint(-1000, 1000, len(age_set))

new_x = np.column_stack((ones, age_set, hgt_set, mnt_set, skl_set, rand_set))
new_y = sal_set

new_x_train = new_x[:80]
new_y_train = new_y[:80]
new_x_test = new_x[80:]
new_y_test = new_y[80:]

coefficients = coefficientCalculation(new_x_train, new_y_train)
new_y_hat = np.dot(new_x_test, coefficients)


print("Test error w/ 80-20 split:")
print("MSE:", mse(new_y_test, new_y_hat), "\n")


coefficients = coefficientCalculation(new_x, new_y)
new_y_hat = np.dot(new_x, coefficients)

print("Training error:")
print("MSE:", mse(y, new_y_hat), "\n")