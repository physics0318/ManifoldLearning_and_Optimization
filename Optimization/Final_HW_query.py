import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Final_HW_class import neuralNetwork, input_nodes, hidden_nodes, output_nodes, learning_rate

img = cv2.imread("3.png", cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap="gray")

img = 255 - img.reshape(784)
img = (img/255*0.99)+0.01

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
trainedwih = pd.read_csv("trained_data_ih.csv")
trainedwho = pd.read_csv("trained_data_oh.csv")


outputs = n.query(img, trainedwih, trainedwho)
ans = np.argmax(outputs)

print(outputs)
print("the predicted number by the neural network is %d" % ans)

plt.show()