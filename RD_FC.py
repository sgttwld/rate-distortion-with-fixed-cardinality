"""
Rate-distortion with fixed cardinality (RDFC) algorithm, Banerjee et al. 2004 
Author: Sebastian Gottwald
Project: https://github.com/sgttwld/rate-distortion-with-fixed-cardinality
"""
import numpy as np
import tensorflow as tf
import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def d(x,y):
    return (tf.expand_dims(x,1)-tf.expand_dims(y,0))**2           

def RDFC_single(X,beta,N):    
    epochs = 10000
    precision = 1e-4

    # tf variable
    init = np.random.uniform(0,1,size=(N))
    q = tf.Variable(init/np.sum(init))
    y = tf.Variable(np.linspace(0,1,N))

    # distortion(Xhat,X)

    t0 = time.time()
    # iterate
    for i in range(epochs):
        expdist = tf.expand_dims(q,0)*tf.exp(-beta*d(X,y))
        post = expdist/tf.expand_dims(tf.reduce_sum(expdist,axis=1),1)
        y.assign(tf.reduce_mean(tf.expand_dims(X,1)*post,axis=0)/tf.reduce_mean(post,axis=0))
        q1 = tf.reduce_mean(post,0)
        if np.linalg.norm(q1.numpy()-q.numpy()) < precision:
            break
        q = q1
    t1 = time.time()

    expdist = tf.expand_dims(q,0)*tf.exp(-beta*d(X,y))
    Z = tf.reduce_sum(expdist,axis=1)
    D = tf.reduce_mean(tf.reduce_sum(expdist*d(X,y),axis=1)/Z)
    R = -beta*D-tf.reduce_mean(tf.math.log(Z))
    return {
        'xhat': y.numpy(),
        'q': q.numpy(), 
        'distortion': D.numpy(), 
        'rate': R.numpy()/np.log(2),
        'episodes': i, 
        'elapsed': t1-t0,
        'beta': beta,
        }

def RDFC(X,beta,N):
    return [RDFC_single(X,b,N) for b in beta]




