import matplotlib.pyplot as plt
import numpy as np

a = []
b = []

# Fitting error model to measurements for CNN
x_cnn = [20, 50, 200, 500, 1000, 2500]
y_cnn = [0.70285714, 0.90285712, 0.98285717, 0.998, 1, 1]

a_vec = np.arange(0, 30, 0.01)
b_vec = np.arange(0, 2, 0.01)
MSE = np.zeros((len(a_vec), len(b_vec)))

a_counter = 0
for aa in a_vec:
    b_counter = 0
    for bb in b_vec:
        MSE_tmp = 0 
        for xx in x_cnn: 
            pred = 1-aa * np.power(xx, -bb)
            tmp = np.square(abs(pred - y_cnn[x_cnn.index(xx)]))
            MSE_tmp = MSE_tmp + tmp

        MSE[a_counter, b_counter] = MSE_tmp
        b_counter += 1
    a_counter += 1

MSE_min = MSE.min()

MSE_min_arg =np.unravel_index(np.argmin(MSE, axis=None), MSE.shape)

a.append(a_vec[MSE_min_arg[0]])
b.append(b_vec[MSE_min_arg[1]])

# Fitting error model to measurements for YOLOV5
x_yolo = [50, 200, 500, 1000, 2500]
y_yolo = [0.285, 0.66, 0.859, 0.89, 0.943]

weights = [0.1, 0.5, 1, 1, 1]
weights = [1, 1, 1, 1, 1]

a_vec = np.arange(0, 10, 0.01)
b_vec = np.arange(0, 2, 0.01)
MSE = np.zeros((len(a_vec), len(b_vec)))

a_counter = 0
for aa in a_vec:
    b_counter = 0
    for bb in b_vec:
        MSE_tmp = 0 
        for xx in x_yolo: 
            pred = 1-aa * np.power(xx, -bb)
            tmp = np.square(abs(pred - y_yolo[x_yolo.index(xx)]))
            # print(b_counter)
            MSE_tmp = MSE_tmp + weights[x_yolo.index(xx)]*tmp

        MSE[a_counter, b_counter] = MSE_tmp
        b_counter += 1
    a_counter += 1

MSE_min = MSE.min()
MSE_min_arg =np.unravel_index(np.argmin(MSE, axis=None), MSE.shape)

a.append(a_vec[MSE_min_arg[0]])
b.append(b_vec[MSE_min_arg[1]])

# Fitting error model to measurements for SECOND
x_second = [50, 200, 500, 1000, 2500]
y_second = [0.659334, 0.709954, 0.715682, 0.721292, 0.792646]


a_vec = np.arange(0, 10, 0.01)
b_vec = np.arange(0, 2, 0.01)
MSE = np.zeros((len(a_vec), len(b_vec)))

a_counter = 0
for aa in a_vec:
    b_counter = 0
    for bb in b_vec:
        MSE_tmp = 0 
        for xx in x_yolo: 
            pred = 1-aa * np.power(xx, -bb)
            tmp = np.square(abs(pred - y_second[x_second.index(xx)]))
            MSE_tmp = MSE_tmp + tmp

        MSE[a_counter, b_counter] = MSE_tmp
        b_counter += 1
    a_counter += 1

MSE_min = MSE.min()
MSE_min_arg =np.unravel_index(np.argmin(MSE, axis=None), MSE.shape)

a.append(a_vec[MSE_min_arg[0]])
b.append(b_vec[MSE_min_arg[1]])

print('The parameters in performance predictors are as follows:')
print('a=', a)
print('b=', b)

# Compute predicted values using performance predictor
x_cnn_pred = [20, 50, 200, 500, 1000, 2500]
y_cnn_pred = 1-a[0] * np.power(x_cnn_pred, -b[0])

x_yolo_pred = [20, 50, 200, 500, 1000, 2500]
y_yolo_pred = 1-a[1] * np.power(x_cnn_pred, -b[1])

x_second_pred = [20, 50, 200, 500, 1000, 2500]
y_second_pred = 1-a[2] * np.power(x_cnn_pred, -b[2])

# Plot the predictions and measurements

plt.xlabel('Number of Samples', fontsize=14)
plt.ylabel('Perception Accuracy', fontsize=14)

plt.plot(x_cnn, y_cnn, 'ro', markerfacecolor = 'none', label='Experimental data, weather classification via CNN')
plt.plot(x_cnn_pred, y_cnn_pred, 'r-', markerfacecolor = 'none', label='Performance predictor, weather classification via CNN')

plt.plot(x_yolo, y_yolo, 'bs', markerfacecolor = 'none', label='Experimental data, sign recognition via YOLOV5')
plt.plot(x_yolo_pred, y_yolo_pred, 'b--', markerfacecolor = 'none', label='Performance predictor, sign recognition via YOLOV5')

plt.plot(x_second, y_second, 'g^', markerfacecolor = 'none',label='Experimental data, object detection via SECOND')
plt.plot(x_second_pred, y_second_pred, 'g-.', markerfacecolor = 'none',label='Performance predictor, object detection via SECOND')

plt.legend(loc='lower right')
plt.show()



