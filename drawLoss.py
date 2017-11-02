import matplotlib.pyplot as plt

base = 'pred_logs/'

loss = open(base+'loss_train.log', 'r')
loss_t = open(base+'loss_test.log', 'r')

loss_data = loss.read().split(' ')[:-1]
loss_data = [float(i) for i in loss_data]

loss_data_t = loss_t.read().split(' ')[:-1]
loss_data_t = [float(i) for i in loss_data_t]

plt.plot(loss_data, 'r-')
loss_train, = plt.plot(loss_data, 'ro')

plt.plot(loss_data_t, 'b-')
loss_test, = plt.plot(loss_data_t, 'bo')
plt.legend([loss_train, loss_test], ['training loss', 'testing loss'])

plt.xlabel('epoch')
plt.ylim(0, 2)
plt.title('loss')
plt.show()

loss.close()
loss_t.close()
