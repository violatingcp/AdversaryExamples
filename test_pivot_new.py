import keras.backend as K
from keras.optimizers import SGD
from keras.layers import Input, Dense, Lambda
from keras.layers.merge import concatenate
from keras.models import Model
from keras.optimizers import Adam 
import matplotlib.pyplot as plt
from IPython import display
plt.rcParams["figure.figsize"] = (6,6)
plt.rcParams["font.size"] = 16.0
import numpy as np
np.random.seed = 333
min_Lr = np.log(1 * (2. * np.pi * np.e) ** 0.5)
from adversarial import Adversary

def turnon(iD,iTrainable,iSkip=-1):
    i0 = -1
    for l in iD.layers:
        i0=i0+1
        if i0 < iSkip:
            continue
        try:
            l.trainable = iTrainable
        except:
            print "trainableErr",layer

def make_X(n_samples, z):
    X0 = np.random.multivariate_normal(mean=np.array([0., 0.]), cov=np.array([[1., -0.5], [-0.5, 1.]]),size=n_samples // 2)
    X1 = np.random.multivariate_normal(mean=np.array([1., 1.]), cov=np.eye(2),size=n_samples // 2)
    X1[:, 1] += z
    X = np.vstack([X0, X1])
    y = np.zeros(n_samples)
    y[n_samples // 2:] = 1
    return X

def plotMe(iD):
    plt.hist(iD.predict(make_X(200000, z=-1)), bins=50, normed=1, histtype="step", label="$p(f(X)|Z=-\sigma)$")
    plt.hist(iD.predict(make_X(200000, z=0)), bins=50, normed=1, histtype="step", label="$p(f(X)|Z=0)$")
    plt.hist(iD.predict(make_X(200000, z=1)), bins=50, normed=1, histtype="step", label="$p(f(X)|Z=+\sigma)$")
    plt.legend(loc="best")
    plt.ylim(0,20)
    plt.xlabel("$f(X)$")
    plt.ylabel("$p(f(X))$")
    plt.grid()
    plt.legend(loc="upper left")
    plt.savefig("f-adversary.pdf")
    plt.show()

def plot_losses(i, losses):
    display.clear_output(wait=True)
    display.display(plt.gcf())

    ax1 = plt.subplot(311)   
    values = np.array(losses["L_f"])
    plt.plot(range(len(values)), values, label=r"$L_f$", color="blue")
    plt.legend(loc="upper right")
    
    ax2 = plt.subplot(312, sharex=ax1) 
    values = np.array(losses["L_r"]) / lam
    plt.plot(range(len(values)), values, label=r"$L_r$", color="green")
    plt.legend(loc="upper right")
    
    ax3 = plt.subplot(313, sharex=ax1)
    values = np.array(losses["L_f - L_r"])
    plt.plot(range(len(values)), values, label=r"$L_f - \lambda L_r$", color="red")  
    plt.legend(loc="upper right")
    
    plt.show()

def make_loss_D(c):
    def loss_D(y_pred, y_true):
        return c * K.binary_crossentropy(y_pred, y_true)
    return loss_D

def make_loss_R(lam, n_components):
    def loss(z_true, z_pred):
        return lam * K.mean(K.square(z_pred - z_true), axis=-1)
    #def loss(y_true, y_pred):
    #    mu = y_pred[:, :n_components]
    #    sigma = y_pred[:, n_components:2*n_components]
    #    pi = y_pred[:, 2*n_components:]
    #    pdf = pi[:, 0] * ((1. / np.sqrt(2. * np.pi)) / sigma[:, 0] *
    #                      K.exp(-(y_true - mu[:, 0]) ** 2 / (2. * sigma[:, 0] ** 2)))
    #    for c in range(1, n_components):
    #        pdf += pi[:, c] * ((1. / np.sqrt(2. * np.pi)) / sigma[:, c] *
    #                           K.exp(-(y_true - mu[:, c]) ** 2 / (2. * sigma[:, c] ** 2)))
    #        nll = -K.log(pdf)
    #    return lam * K.mean(nll)
    
    return loss


n_samples = 125000

X0 = np.random.multivariate_normal(mean=np.array([0., 0.]), cov=np.array([[1., -0.5], [-0.5, 1.]]),size=n_samples // 2)
X1 = np.random.multivariate_normal(mean=np.array([1., 1.]), cov=np.eye(2),size=n_samples // 2)
z  = np.random.normal(loc=0.0,scale=1.0,size=n_samples)
X1[:, 1] += z[n_samples // 2:]

X = np.vstack([X0, X1])
y = np.zeros(n_samples)
y[n_samples // 2:] = 1


#plt.title("$X$")
#plt.scatter(X[y==0, 0], X[y==0, 1], c="r", marker="o", edgecolors="none")
#plt.scatter(X[y==1, 0], X[y==1, 1], c="b", marker="o", edgecolors="none")
#plt.xlim(-4, 4)
#plt.ylim(-4, 4)
#plt.show()
#a

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid, z_train, z_valid = train_test_split(X, y, z, test_size=50000)

inputs = Input(shape=(X.shape[1],))
Dx = Dense(20, activation="tanh")(inputs)
Dx = Dense(20, activation="relu")(Dx)
Dx = Dense(1, activation="sigmoid")(Dx)
D  = Model(inputs=[inputs], outputs=[Dx])
opt_D = SGD()
D.compile(loss=[make_loss_D(c=1.0)], optimizer=opt_D)
D.summary()

lam = 100.0 
#lam = 1000.0 
n_components = 5
Rx  = Adversary(n_components,1,scale=1.0)(D(inputs))
R   = Model(inputs=[inputs], outputs=[Rx])
RT  = Model(inputs=[inputs], outputs=[D(inputs),R(inputs)])
opt_DfR = SGD(momentum=0.0)
RT.compile(loss=[make_loss_D(c=1.0),make_loss_R(lam, n_components)], optimizer=Adam(lr=0.00001),metrics=['accuracy'])#optimizer=opt_DfR)
RT.summary()

# Pretraining of D
D.fit(X_train, y_train, epochs=10)
plotMe(D)
min_Lf = D.evaluate(X_valid, y_valid)
RT.fit(X_train,[y_train,z_train],epochs=20)
plotMe(D)
