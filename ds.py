import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import os
from imageio import imread
import imageio.v2 as imageio
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input

def scale(img):
    return img / 255.

pwd = os.getcwd()
path = os.path.join(pwd,"extracted_images")
dir_list = os.listdir(path = path)

X_full=[]
Y_full=[]
label_list=[]

i=0

for dir in dir_list:
    if(dir[0]=="."): continue
    path1 = os.path.join(path,dir)
    allfiles = os.listdir(path1)
    print(dir)
    label_list.append(dir[0])
    j=0

    for file in allfiles:
        if j > 150: break
        t1 = os.path.join(path1,file)
        a1 = imageio.imread(t1).flatten()
        X_full.append(scale(a1))
        Y_full.append(i)
        j+=1

    i+=1
XX=np.array(X_full)
YY=np.array(Y_full)

plt.imshow(XX[1].reshape(45,45),cmap=plt.cm.gray)

## Split images
X_train,X_test,y_train,y_test = train_test_split(XX,YY,random_state=77,test_size=.15)

sizeInput = X_train.shape[1]
OutputVector = 82
sam = np.eye(OutputVector)

train_label=sam[y_train.flatten()-1]
test_label =sam[y_test.flatten()-1]

train_label.shape

### CNN Convolutional Neural Networks
class Model:

    def __init__(self,sess,name):
        self.sess = sess
        self.name = name
        self._build_net()

    def _build_net(self):

        with tf.variable_scope(self.name):

            self.learning_rate=0.7
            self.training = tf.placeholder(tf.bool)

            ### placeholder
            self.X = tf.placeholder(tf.float32,[None,sizeInput])
            self.Y = tf.placeholder(tf.float32,[None,OutputVector])
            self.X_img = tf.reshape(self.X,[-1,45,45,1])

            ## conv layer 1 -> 5*5*32 filter
            ## dropout with ratio 0.7

            with tf.variable_scope('layer1'):
                conv1 = tf.layers.conv2d(self.X_img,filters=32,kernel_size=5,strides=1,
                                         padding="SAME",activation=tf.nn.relu)

                pool1 = tf.layers.max_pooling2d(conv1,pool_size=2,strides=2,padding="SAME")

                dropout1 = tf.layers.dropout(pool1,rate=0.7,training=self.training)

            ## conv layer 2 -> 3*3*64 filters
            ## dropout with ratio 0.7

            with tf.variable_scope('layer2'):
                conv2 = tf.layers.conv2d(dropout1,filters=64,kernel_size=3,strides=1,
                                        padding="SAME",activation=tf.nn.relu)

                pool2 = tf.layers.max_pooling2d(conv2,pool_size=2,strides=2,padding="SAME")
                dropout2 = tf.layers.dropout(pool2,rate=0.7,training=self.training)

            ## conv layer 3 -> 3*3*128 filters
            ## dropout with ratio 0.7

            with tf.variable_scope('layer3'):
                conv3 = tf.layers.conv2d(dropout2,filters=128,kernel_size=3,strides=1,
                                        padding="SAME",activation=tf.nn.relu)
                pool3 = tf.layers.max_pooling2d(conv3,pool_size=2,strides=2,padding="SAME")
                dropout3 = tf.layers.dropout(pool3,rate=0.7,training=self.training)

            s1 = int(dropout3.shape[1])

            with tf.variable_scope('layer4'):
                flat = tf.reshape(dropout3,[-1,s1*s1*128])
                f4 = tf.layers.dense(flat,units=1024,activation=tf.nn.relu)
                dropout4 = tf.layers.dropout(f4,rate=0.5,training=self.training)

            with tf.variable_scope('output'):
                self.logits = tf.layers.dense(dropout4,units=OutputVector)

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.Y))
        self.optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(self.cost)

        correct_prediction=tf.equal(tf.argmax(self.logits,1),tf.argmax(self.Y,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    def predict(self,X_test,training=False):
        return self.sess.run(self.logits,
                            feed_dict={self.X:X_test,self.training:training})

    def get_accuracy(self,X_test,Y_test,training=False):
        return self.sess.run(self.accuracy,
                            feed_dict={self.X:X_test,self.Y:Y_test,self.training:training})

    def train(self,x_data,y_data,training=True):
        return self.sess.run([self.cost,self.optimizer],
                            feed_dict={self.X:x_data,self.Y:y_data,self.training:training})

model = Sequential([
    Input(shape=(2025,)),
    Dense(128, activation='relu'),
    Dense(82, activation='softmax')
])


# Compilare model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Hiperparametrii
batch_size = 200
train_epoch = 50

print("Start Training")

for epoch in range(train_epoch):
    p1 = np.random.permutation(len(X_train))
    x_data = X_train[p1]
    y_data = train_label[p1]

    p2 = np.random.permutation(len(X_test))
    x_test = X_test[p2]
    y_test = test_label[p2]

    avg_cost = 0.0

    num_batch = int(len(X_train) / batch_size)

    for i in range(num_batch):
        batch_xs, batch_ys = x_data[i*batch_size:((i+1)*batch_size)], y_data[i*batch_size:((i+1)*batch_size)]
        # Antrenează modelul pe batch
        model.fit(batch_xs, batch_ys, epochs=1, batch_size=batch_size, verbose=0)

    training_acc_cnn = model.evaluate(x_data, y_data, verbose=0)
    test_acc_cnn = model.evaluate(x_test, y_test, verbose=0)

    print(f'Epoch: {epoch+1}, Training Accuracy: {training_acc_cnn[1]}, Test Accuracy: {test_acc_cnn[1]}')

print('Learning Finished')

model.save('trained_model.h5')  # Salvează modelul complet într-un fișier .h5

print("Model saved successfully!")

# plt.plot(test_acc_reg,'bo',label="test_reg")
# plt.plot(test_acc_reg,'b--')
plt.plot(test_acc_cnn,'ro',label="test_cnn")
plt.plot(test_acc_cnn,'r--')
plt.ylabel('Accuracy')  # Accuracy label
plt.xlabel('Epoch')
plt.title('Plot of Test Accuracies Reg vs CNN')
plt.legend()
ax = plt.gca()
ymin, ymax = ax.get_ylim()
xmin, xmax = ax.get_xlim()
midXLoc = (xmax - xmin)/2
topYLoc = ymax - (ymax-ymin)*0.7
plt.text(midXLoc, topYLoc, 'Num Epochs: ' + str(train_epoch) + '\nBatch Size: ' + str(batch_size), ha='center', va='top')

plt.savefig('Code2_test_error', dpi=300)

np.max(test_acc_cnn)