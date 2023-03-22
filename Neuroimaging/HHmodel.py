import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
from tqdm import tqdm
class Neuron():
    def __init__(self,args:Dict):
        for k,v in args.items():
            setattr(self, k, v)
        self.V = self.Vrest
        self.I = 0
        self.Vlist = []
        self.nlist = []
        self.mlist = []
        self.hlist = []
        self.Ilist = []

    def update(self):

        alphan = 0.01*(-self.V+10)/(np.exp((-self.V+10)/10)-1)
        betan = 0.125*np.exp(-self.V/80)
        alpham = 0.1*(-self.V+25)/(np.exp((-self.V+25)/10)-1)
        betam = 4*np.exp(-self.V/18)
        alphah = 0.07*np.exp(-self.V/20)
        betah = 1/(np.exp((-self.V+30)/10)+1)

        dn = (alphan * (1 - self.n) - betan * self.n) * self.dt
        dm = (alpham * (1 - self.m) - betam * self.m) * self.dt
        dh = (alphah * (1 - self.h) - betah * self.h) * self.dt
        self.n = self.n + dn
        self.m = self.m + dm
        self.h = self.h + dh

        dV = self.dt * (self.I - self.gK * (self.n**4) * (self.V-self.VK) - self.gNa * (self.m**3) * self.h * (self.V-self.VNa)\
            - self.gl * (self.V - self.Vl))/self.cM

        self.V = self.V + dV

        self.Vlist.append(self.V)
        self.nlist.append(self.n)
        self.mlist.append(self.m)
        self.hlist.append(self.h)
        self.Ilist.append(self.I)

    # def update_mouse(self):
    #     alphan = 0.182*(self.V+35)/(1-np.exp((-self.V-35)/9))
    #     betan = -0.124*(self.V+35)/(1-np.exp((self.V+35)/9))
    #     alpham = 0.02*(self.V-25)/(1-np.exp((-self.V+25)/9))
    #     betam = -0.002*(self.V-25)/(1-np.exp((self.V-25)/9))
    #     alphah = 1/(1+np.exp(-(self.V+62)/6))
    #     betah = 4*np.exp((self.V+90)/12)/(1+np.exp((-(self.V+62))/6))

    #     dn = (alphan * (1 - self.n) - betan * self.n) * self.dt
    #     dm = (alpham * (1 - self.m) - betam * self.m) * self.dt
    #     dh = (alphah * (1 - self.h) - betah * self.h) * self.dt
    #     self.n = self.n + dn
    #     self.m = self.m + dm
    #     self.h = self.h + dh

    #     dV = self.dt * (self.I - self.gK * (self.n**4) * (self.V-self.VK) - self.gNa * (self.m**3) * self.h * (self.V-self.VNa)\
    #         - self.gl * (self.V - self.Vl))/self.cM

    #     self.V = self.V + dV

    #     self.Vlist.append(self.V)
    #     self.nlist.append(self.n)
    #     self.mlist.append(self.m)
    #     self.hlist.append(self.h)
    #     self.Ilist.append(self.I)


    def rate(self,Vlist):
        i = 0
        count = 0
        threshold = 70
        while i < len(Vlist):
            if Vlist[i] > threshold:
                count += 1
                for j in range(0, len(Vlist)-i):
                    if Vlist[i+j]<Vlist[i]:
                        break
                if j!=0:
                    i += j
                else:
                    break
            i += 1
        return count/self.duration


    def run(self,t,I_in):
        step = int(t//self.dt)
        self.I_in = I_in
        for i in range(step):
            if i*self.dt > self.start:
                self.I = self.I_in
            if i*self.dt > self.duration:
                self.I = 0
            self.update()
        return self.Vlist, self.nlist, self.mlist, self.hlist, self.Ilist, self.rate(self.Vlist)

neuron = Neuron({
    'Vrest':0,
    'start':15,
    'duration':60,
    'cM':1,
    'VNa':115,
    'VK':-12,
    'Vl':10.613,
    'gNa':120,
    'gK':36,
    'gl':0.3,
    'dt':0.05,
    'n':0.3,
    'm':0,
    'h':0.6,
    'mouse':0
})
        
#这一部分用来跑rate-current关系
# Iarr = np.linspace(0,150,51)

# rlist = []
# print(Iarr)
# for i in tqdm(range(len(Iarr))):
#     V,n,m,h,Ilist,rate = neuron.run(80,Iarr[i])
#     rlist.append(rate)
# plt.plot(Iarr, rlist)
# plt.ylabel('rate/s^{-1}')
# plt.xlabel('current/uA')
# plt.show()

#这一部分用来跑单个神经元活动
Iarr = [5,10,30]
V,n,m,h,Ilist,rate = neuron.run(80,5)
plt.subplot(311)
plt.plot(V)
plt.ylabel('V/mV')
plt.subplot(312)
plt.plot(n)
plt.plot(m)
plt.plot(h)
plt.legend(['n','m','h'])
plt.subplot(313)
plt.plot(Ilist)
plt.ylabel('I/uA')
plt.show()
