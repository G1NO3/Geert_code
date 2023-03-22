from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

Y = np.array([1]*8 + [-1]*9)
X = np.array([.697,.46,
    .774,.376,
    .634,.264,
    .608,.318,
    .556,.215,
    .403,.237,
    .481,.149,
    .437,.211,
    .666,.091,
    .243,.267,
    .245,.057,
    .343,.099,
    .639,.161,
    .657,.198,
    .36,.37,
    .593,.042,
    .719,.103]).reshape(-1,2)
model = svm.SVC(C=5,kernel='linear')
model.fit(X,Y)
y_pred = model.predict(X)
print(y_pred)
plt.scatter(X[:,0],X[:,1],c=Y)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[-1,0,1], alpha=0.5,
           linestyles=['--', '-', ':'])

ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')

plt.show()