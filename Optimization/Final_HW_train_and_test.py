import numpy as np
import time
import pandas as pd
from Final_HW_class import neuralNetwork, input_nodes, hidden_nodes, output_nodes

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes)

training_data_file = open("mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 1
sec1 = time.time()
for e in range(epochs):
    train_counter = 0
    for record in training_data_list:
        train_counter += 1
        all_values = record.split(',')

        inputs = (np.asfarray(all_values[1:])/255.0*0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets, train_counter)
        if train_counter % 1000 == 0:
            print("%d-th training is over, e = %d" % (train_counter, e+1))
        continue
    pd.DataFrame(n.wih).to_csv("trained_data_ih.csv", index=False)
    pd.DataFrame(n.who).to_csv("trained_data_oh.csv", index=False)
    pass

sec2 = time.time()
print("training is over")

test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

trainedwih = pd.read_csv("trained_data_ih.csv")
trainedwho = pd.read_csv("trained_data_oh.csv")
       
scorecard = []
for record in test_data_list:

    all_values = record.split(',')
    correct_label = int(all_values[0])

    inputs = (np.asfarray(all_values[1:])/255.0*0.99) + 0.01
    outputs = n.query(inputs, trainedwih, trainedwho)
    label = np.argmax(outputs)

    scorecard.append(1) if (label == correct_label) else scorecard.append(0)
    pass

sec3 = time.time()

print("\n\ntraining time : %fsec" % (sec2-sec1))
print("testing time : %fsec" % (sec3-sec2))

scorecard_array = np.asarray(scorecard)
print("performance : ", scorecard_array.sum()/scorecard_array.size)
