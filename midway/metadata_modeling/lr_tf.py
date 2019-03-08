#%%
import numpy as np
import os
import sys
import tensorflow as tf

''' global definitions '''
CMND_LINE_ENABLED = False
LEARNING_RATE = 0.1
MOMENTUM = 0.9
EPOCHS = 150
BATCH_SIZE = 32
SEED = 1
DATA_PATH = f'{os.getcwd()}/data/'
RESULTS_PATH = f'{os.getcwd()}/results/'
MODELS_PATH = f'{os.getcwd()}/results/'
CLASSES = 58
ARGVS = [
    'NULL',                                       
    DATA_PATH + f'train_brand_design_mtx_{CLASSES}.npy',     
    DATA_PATH + f'train_one_hot_labels_{CLASSES}.npy',
    DATA_PATH + f'test_brand_design_mtx_{CLASSES}.npy',    
    DATA_PATH + f'test_one_hot_labels_{CLASSES}.npy',
    RESULTS_PATH + f'lr_brand_by_epoch_{CLASSES}.npy',
    MODELS_PATH + f'lr_brand_{CLASSES}'
]

argvs = sys.argv if CMND_LINE_ENABLED else ARGVS
train_design_mtx = np.load(argvs[1])
train_one_hot = np.load(argvs[2])
test_design_mtx = np.load(argvs[3])
test_one_hot = np.load(argvs[4])
_, n_features = train_design_mtx.shape
_, n_classes = train_one_hot.shape


#%%
# graph input
x = tf.placeholder(tf.float32,[None,n_features])
y = tf.placeholder(tf.float32,[None,n_classes])
w = tf.Variable(tf.random.uniform(shape =[n_features,n_classes],minval=-1,maxval=1,seed=SEED))
b = tf.Variable(tf.zeros([n_classes]))
w

# model
dense = tf.matmul(x,w) + b
softmax = tf.nn.softmax(dense)

# loss and optimizer
loss = tf.losses.softmax_cross_entropy(onehot_labels = y,logits = dense)
optimizer = tf.train.MomentumOptimizer(learning_rate = LEARNING_RATE,
                                       momentum = MOMENTUM).minimize(loss)

# evaluation
predictions, actuals = tf.math.argmax(softmax,axis=1), tf.math.argmax(y,axis=1)
correct = tf.cast(tf.equal(predictions,actuals),dtype=tf.float64)
percent_error = 1 - tf.reduce_mean(correct)

# begin session
init = tf.global_variables_initializer()
with tf.Session() as sess:
    results = np.zeros((EPOCHS,4)) # store epoch results in here
    saver = tf.train.Saver() 
    sess.run(init)
    for epoch in range(EPOCHS):
        # shuffle data
        x_epoch, y_epoch = train_design_mtx.copy(), train_one_hot.copy()
        idx = np.arange(len(x_epoch))
        np.random.shuffle(idx)
        x_epoch, y_epoch = x_epoch[idx], y_epoch[idx]
        while np.size(x_epoch):
            x_batch, y_batch = x_epoch[0:BATCH_SIZE], y_epoch[0:BATCH_SIZE]
            sess.run(optimizer,feed_dict={x: x_batch,y: y_batch})
            x_epoch, y_epoch = x_epoch[BATCH_SIZE:], y_epoch[BATCH_SIZE:]
        
        # print evaluations
        train_loss = sess.run(loss,feed_dict={x: train_design_mtx, y: train_one_hot})
        test_loss = sess.run(loss,feed_dict={x: test_design_mtx, y: test_one_hot})
        train_perc_error = sess.run(percent_error,feed_dict={x: train_design_mtx, y: train_one_hot})
        test_perc_error = sess.run(percent_error,feed_dict={x:test_design_mtx, y: test_one_hot})
        print(epoch,train_loss,test_loss,train_perc_error,test_perc_error)
        results[epoch,:] = np.array([train_loss,test_loss,train_perc_error,test_perc_error])
    
    # store results and model
    saver.save(sess,argvs[6],global_step=epoch)
    np.save(argvs[5],results)



    

        

















