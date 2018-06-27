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

def turnon(iD,iTrainable,iOther=0):#iSkip=-1):
    i0 = -1
    for l1 in iD.layers:
        i0=i0+1
        if iOther != 0 and l1 in iOther.layers:
            continue
        try:
            l1.trainable = iTrainable
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
        #y_true = y_true[0]#.ravel()
    #    mu = y_pred[:, :n_components]
    #    sigma = y_pred[:, n_components:2*n_components]
    #    pi = y_pred[:, 2*n_components:]
    #    pdf = pi[:, 0] * ((1. / np.sqrt(2. * np.pi)) / sigma[:, 0] *
    #                      K.exp(-(y_true - mu[:, 0]) ** 2 / (2. * sigma[:, 0] ** 2)))
    #    for c in range(1, n_components):
    #        pdf += pi[:, c] * ((1. / np.sqrt(2. * np.pi)) / sigma[:, c] *
    #                           K.exp(-(y_true - mu[:, c]) ** 2 / (2. * sigma[:, c] ** 2)))
    #        nll = -K.log(pdf)
    #    return lam * nll#K.mean(nll)
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
D = Model(inputs=[inputs], outputs=[Dx])

n_components = 5
Rx    = Dx
Rx    = Dense(20, activation="relu")(Rx)
Rx    = Dense(20, activation="relu")(Rx)
mu    = Dense(n_components, activation="linear",name='a') (Rx)
sigma = Dense(n_components, activation=K.exp,name='b')    (Rx)
pi    = Dense(n_components, activation="softmax",name='c')(Rx)
Rx    = concatenate([mu, sigma, pi])
#Rx   = Merge(mode="concat")([mu, sigma, pi])
R     = Model(inputs=[inputs], outputs=[Rx])


#lam = 50000.0 
lam = 50.0 
turnon(D,False)        
#turnon(R,True,2)
opt_DfR = SGD(momentum=0.0)
#opt_DfR = Adam()
DfR = Model(inputs=[inputs], outputs=[R(inputs)])
DfR.compile(loss=[make_loss_R(1.0, n_components)], optimizer=opt_DfR)

turnon(R,False,D)    
turnon(D,True)                                                                                                                                                                                                                                                         
opt_D = SGD()
D.compile(loss=[make_loss_D(c=1.0)], optimizer=opt_D)
opt_DRf = SGD(momentum=0.0)
DRf = Model(inputs=[inputs], outputs=[D(inputs), R(inputs)])
DRf.compile(loss=[make_loss_D(c=1.0), make_loss_R(-lam, n_components)], optimizer=opt_DRf)
DRf.summary()

# Pretraining of D
turnon(R,False,D)
turnon(D,True)
D.fit(X_train, y_train, epochs=10)
D.summary()
turnon(D,False)
turnon(R,True,D)
DfR.summary()
DfR.fit(X_train, z_train, epochs=10)
plotMe(D)
min_Lf = D.evaluate(X_valid, y_valid)
batch_size = 1280
losses = {"L_f": [], "L_r": [], "L_f - L_r": []}
for i in range(21):
    l = DRf.evaluate(X_valid, [y_valid, z_valid], verbose=0)    
    losses["L_f - L_r"].append(l[0][None][0])
    losses["L_f"].append(l[1][None][0])
    losses["L_r"].append(-l[2][None][0])
    #print(losses["L_r"][-1] / lam)
    print l[0],l[1],l[2]
    #if i % 500 == 0:
    #    plot_losses(i, losses)

    # Fit D
    turnon(R,False,D)
    turnon(D,True)
    indices = np.random.permutation(len(X_train))[:batch_size]
    #DRf.fit(X_train, [y_train, z_train],batch_size=batch_size,nb_epoch=1,verbose=1)
    #DRf.fit(X_train[indices], [y_train[indices], z_train[indices]],epochs=1,verbose=1)
    DRf.train_on_batch(X_train[indices], [y_train[indices], z_train[indices]])
    #D.summary()
    #DRf.summary()
    # Fit R
    turnon(D,False)
    turnon(R,True,D)
    #DfR.summary()
    indices = np.random.permutation(len(X_train))[:batch_size]
    DfR.train_on_batch(X_train[indices], z_train[indices])#, batch_size=batch_size, nb_epoch=1, verbose=1)
    #DfR.fit(X_train[indices], z_train[indices],epochs=1,verbose=1)

plotMe(D)
