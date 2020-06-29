import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def generate_data(num_samples, num_features):
    #Randomly generates a number of data points.
    data_size = (num_samples,num_features)
    data = np.random.randint(0,100, size=data_size)
    labels_size = (num_samples, 1)
    labels = np.random.randint(0, 2, size=labels_size)
    return data.astype(np.float32), labels

train_data, labels = generate_data(11,2)

print(train_data)
print(labels)

plt.plot(train_data[0, 0], train_data[0, 1], 'sb')
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')
plt.show()