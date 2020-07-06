import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors as skn

def generate_data(num_samples, num_coord , num_labels):
    #Randomly generates a number of data points.
    data_size = (num_samples,num_coord)
    data = np.random.randint(0,100, size=data_size)
    labels_size = (num_samples, 1)
    labels = np.random.randint(0, num_labels, size=labels_size)
    return data.astype(np.float32), labels

train_data, labels = generate_data(11,2,2)

all_blue = train_data[labels.ravel()==0]
all_red = train_data[labels.ravel()==1]
print(all_blue)

plt.scatter(all_blue[:, 0], all_blue[:, 1], c='b', marker='s', s=180)
plt.scatter(all_red[:, 0], all_red[:, 1], c='r', marker='^', s=180)

plt.xlabel('x coordinate (feature 1)')
plt.ylabel('y coordinate (feature 2)')

newCase, _ = generate_data(1,2,2)

plt.plot(newCase[0, 0], newCase[0, 1], 'go', markersize=14)
plt.show()

model = skn.KNeighborsClassifier(7)

model.fit(train_data,labels)

result = model.predict(newCase)
print(result)