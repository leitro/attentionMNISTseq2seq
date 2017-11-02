import matplotlib.pyplot as plt

base = 'pred_logs/'

cer = open(base+'cer_train.log', 'r')
cer_data = cer.read().split(' ')[:-1]
cerr = [float(i) for i in cer_data]

cer2 = open(base+'cer_train2.log', 'r')
cer_data2 = cer2.read().split(' ')[:-1]
cerr2 = [float(i) for i in cer_data2]

cer_t = open(base+'cer_test.log', 'r')
cer_data_t = cer_t.read().split(' ')[:-1]
cerr_t = [float(i) for i in cer_data_t]

plt.plot(cerr, 'r-')
cer_spot, = plt.plot(cerr, 'ro')

plt.plot(cerr2, 'c-')
cer_spot2, = plt.plot(cerr2, 'co')

plt.plot(cerr_t, 'b-')
cer_spot_t, = plt.plot(cerr_t, 'bo')
plt.legend([cer_spot, cer_spot2, cer_spot_t], ['CER train with true label', 'CER train with predicted label', 'CER test'])
plt.xlabel('epoch')
plt.ylim(0, 1.1)
plt.title('character error rate')
plt.show()

cer.close()
cer2.close()
cer_t.close()
