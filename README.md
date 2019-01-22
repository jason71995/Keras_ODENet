# Neural Ordinary Differential Equations in Keras

## Introduction

Implementation of [(2018) Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366).

## Attention

ODE solver are use [tf.contrib.integrate.odeint](https://www.tensorflow.org/api_docs/python/tf/contrib/integrate/odeint) which only supported "dopri5" method now.

## Environment

```
python==3.6
tensorflow==1.4.0
keras==2.1.0
```

## Result

Result on 10 epochs

### MNIST ODENet
train_loss: 0.0107 - train_acc: 0.9965 - val_loss: 0.0295 - val_acc: 0.9929

### MNIST ResNet
train_loss: 0.0096 - train_acc: 0.9968 - val_loss: 0.0307 - val_acc: 0.9908

