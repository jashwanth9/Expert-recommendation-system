import matplotlib.pyplot as plt
import numpy as np

f = open('training_stats.txt', 'r')
data = f.readlines()
f.close()

loss = []
for i in data:
	i = i.replace('\n', '')
	i = i.split()
	loss.append(float(i[-1]))
loss = np.array(loss)

plt.plot(np.arange(1,loss.shape[0]+1), loss)
plt.xlabel('# Epochs')
plt.ylabel('Loss')
plt.title('5 layer net with 512 neurons in each hidden layer')
plt.show()