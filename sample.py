#optimization method
optimizer = optimizers.MomentumSGD(momentum = .99)
#optimizer = optimizers.RMSprop(lr=0.001, alpha=0.99, eps=1e-08)
optimizer.setup(model)


#learning rate
optimizer.lr = .01

#dropout probability
p = .25

#number of training epochs
num_epochs = 400

L_Y_train = len(y_train)

time1 = time.time()
for epoch in range(num_epochs):
    #reshuffle dataset
    I_permutation = np.random.permutation(L_Y_train)
    x_train = x_train[I_permutation,:]
    y_train = y_train[I_permutation]
    epoch_accuracy = 0.0
    batch_counter = 0
    for i in range(0, L_Y_train, batch_size):
        if (GPU_on):
            x_batch = cuda.to_gpu(x_train[i:i+batch_size,:])
            y_batch = cuda.to_gpu(y_train[i:i+batch_size] )
        else:
            x_batch = x_train[i:i+batch_size,:]
            y_batch = y_train[i:i+batch_size]
        model.zerograds()
        dropout_bool = True
        bn_bool = False
        loss, accuracy = model(x_batch, y_batch, dropout_bool, bn_bool, p)
        loss.backward()
        optimizer.update()
        #print("success")
        epoch_accuracy += np.float(accuracy.data)
        batch_counter += 1
    if (epoch % 1 == 0):
        #print "Epoch %d" % epoch
        train_accuracy = 100.0*epoch_accuracy/np.float(batch_counter)
        print "Train accuracy: %f" % train_accuracy
    if (epoch == num_epochs-1):
        test_accuracy = Calculate_Test_Accuracy(x_test, y_test, model, p, GPU_on, batch_size)
        print "Test Accuracy: %f" % test_accuracy


time2 = time.time()
training_time = time2 - time1
print "Rank: %d" % rank
print "Training time: %f" % training_time

