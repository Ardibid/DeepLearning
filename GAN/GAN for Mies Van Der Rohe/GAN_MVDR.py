# based on code from:
# https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/GAN/dcgan_mnist.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout,BatchNormalization,Activation, Reshape, LeakyReLU, Flatten
from keras.layers import Conv2DTranspose,Conv2D,UpSampling2D
from keras.optimizers import adam, RMSprop


from os import listdir
from os.path import isfile, join
import cv2

import time
from time import gmtime, strftime
import imageio


def readImageNames(path = None):
    if path == None:
        path = "./dataSet/"
    allFiles = [f for f in listdir(path) if isfile(join(path, f))]
    return [img for img in allFiles if img[-4:] == ".jpg"]

def readImages(names, path,imgRows, imgCols,channels):
    if path == None:
        path = "./"
    newPath = path+"resized/"
    imageList = []
    counter = 0
    first = True
    for name in names:        
        tmpPath = join(path, name)
        if channels == 1:
            img = misc.imread(tmpPath,'F')
        else: 
            img = misc.imread(tmpPath)
        img = misc.imresize(img, (imgRows,imgCols))
        img = img.astype(np.float32)
        img /= 255.
        #img = img.ravel()
        img = np.reshape(img, (imgRows, imgCols,channels))
        imageList.append(img)
    return np.array(imageList)

def showImage(img):
    print("Image shape: {}".format(img.shape))
    plt.imshow(img)
    plt.show()
    return None

def processImages(imgRows, imgCols , channels):
    # reads images from the source directory as black and white with range of [0.0-1]
    # the output will be a numpy array of shape: [number of images, size*size]
    names = readImageNames()
    images = readImages(names,"./resized/",imgRows, imgCols, channels = channels)
    print ("{} images loaded".format(len(images)))
    return images



class GAN(object):
    
    def __init__(self,imgRows = 28, imgCols = 28, channels = 1):
        # Hyper parameters
        self.imgRows = imgRows
        self.imgCols = imgCols
        self.channels = channels
        
        # Models 
        self.Dis = None
        self.Gen = None
        self.AdvModel = None
        self.DisModel = None
        
    def discriminator(self):
        if self.Dis:
            return self.Dis
        # hyper parameters
        depth = 64
        dropOutRate = 0.4
        
        # model architecture
        self.Dis = Sequential()
        imputShape = (self.imgRows, self.imgCols, self.channels)
        
        # add layers
        self.Dis.add(Conv2D(depth*1, 5, strides=2, input_shape=imputShape,padding='same'))
        self.Dis.add(LeakyReLU(alpha=0.2))
        self.Dis.add(Dropout(dropOutRate))

        self.Dis.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.Dis.add(LeakyReLU(alpha=0.2))
        self.Dis.add(Dropout(dropOutRate))

        self.Dis.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.Dis.add(LeakyReLU(alpha=0.2))
        self.Dis.add(Dropout(dropOutRate))

        self.Dis.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.Dis.add(LeakyReLU(alpha=0.2))
        self.Dis.add(Dropout(dropOutRate))
        
        self.Dis.add(Flatten())
        self.Dis.add(Dense(1))
        
        
        # wrapping up
        self.Dis.add(Activation('sigmoid'))
        self.Dis.summary()
        return self.Dis
        
        
    def generator(self):
        
        if self.Gen:
            return self.Gen
        
        self.Gen = Sequential()
        dropOutRate = 0.4
        depth = 64+64+64+64
        dim = 7
        
        # add layers
        self.Gen.add(Dense(dim*dim*depth, input_dim=100))
        self.Gen.add(BatchNormalization(momentum=0.9))
        self.Gen.add(Activation('relu'))
        self.Gen.add(Reshape((dim, dim, depth)))
        self.Gen.add(Dropout(dropOutRate))

        self.Gen.add(UpSampling2D())
        self.Gen.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.Gen.add(BatchNormalization(momentum=0.9))
        self.Gen.add(Activation('relu'))

        self.Gen.add(UpSampling2D())
        self.Gen.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.Gen.add(BatchNormalization(momentum=0.9))
        self.Gen.add(Activation('relu'))

        self.Gen.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.Gen.add(BatchNormalization(momentum=0.9))
        self.Gen.add(Activation('relu'))

        self.Gen.add(Conv2DTranspose(1, 5, padding='same'))
        
        # wrapping up
        self.Gen.add(Activation('sigmoid'))
        self.Gen.summary()
        return self.Gen
    
    def discriminatorModel(self):
        if self.DisModel:
            return self.DisModel
        
         # hyper paramters
        lr = 0.0002
        decay = 6e-8
        
        # architecture
        optimizer = RMSprop(lr = lr, decay = decay)
        self.DisModel = Sequential()
        self.DisModel.add(self.discriminator())
        self.DisModel.compile(loss='binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])
        
        return self.DisModel
        
    def adversarialModel(self):
        if self.AdvModel:
            return self.AdvModel
        
        # hyper paramters
        lr = 0.0001
        decay = 3e-8
        
        # architecture
        optimizer = RMSprop(lr = lr, decay = decay)
        self.AdvModel = Sequential()
        self.AdvModel.add(self.generator())
        self.AdvModel.add(self.discriminator())
        self.AdvModel.compile(loss='binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])
        
        return self.AdvModel


class MVDRGAN(object):
    def __init__(self):
        # Hyper parameters
        self.imgRows = 28
        self.imgCols = 28
        self.channels = 1
        
        # read data
        self.X= processImages(imgRows=self.imgRows, imgCols = self.imgCols, channels = self.channels)
        
        self.MVDRGAN = GAN()
        self.discriminator = self.MVDRGAN.discriminatorModel()
        self.adversarial= self.MVDRGAN.adversarialModel()
        self.generator = self.MVDRGAN.generator()
        
    def train(self, train_steps=2000, batch_size=64, save_interval=0):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):
            startTime = time.time()
            images_train = self.X[np.random.randint(0,self.X.shape[0], size=batch_size), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)
            #print (images_train.shape, images_fake.shape)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            #print (x.shape, y.shape)
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            duration = int(1000*(time.time() - startTime))
            log_mesg = "%s [Time: %d ms]" %(log_mesg, duration)
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))
        self.makeGIF(train_steps,save_interval)

                    
    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'MVDR.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "./generated/MVDR_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.X.shape[0], samples)
            images = self.X[i, :, :, :]

        plt.figure(figsize=(28,28))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.imgRows, self.imgCols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

    def makeGIF(self,train_steps,save_interval):
        filenames = []
        for i in range(1,train_steps//save_interval):
            filenames.append("./generated/MVDR_{}.png".format(i*save_interval))
        fileName = strftime("./gif/movie_%H_%M_%S.gif", gmtime())
        with imageio.get_writer(fileName, mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

if __name__ == "__main__":
    MVDRGAN = MVDRGAN()
    MVDRGAN.train(train_steps=30000, batch_size=64, save_interval=50)
    MVDRGAN.plot_images(fake=True)
    MVDRGAN.plot_images(fake=False, save2file=True)
