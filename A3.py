import numpy as np
import matplotlib.pyplot as plt
import math
mu, sigma = 0, 0.1 # mean and standard deviation
x=[1,60.18, 86.30]
thetas=[]
log_costs=[]
lin_costs=[]
hyps=[]
x=np.array(x)
for i in range(1000):
    theta = np.random.normal(mu, sigma, 3)
    thetas.append(theta)
    theta=np.array(theta)
    theta=theta.transpose()
    thetaX = np.matmul(theta, x)
    hyp=1/(1+(math.e)**((-1)*thetaX))
    hyps.append(hyp)
    log_cost=math.log(hyp) * (-1)
    lin_cost=(1-hyp**2)/2
    #print theta.shape,x.shape
    log_costs.append(log_cost)
    lin_costs.append(lin_cost)
plt.subplot(2,1,1)
plt.plot(hyps, lin_costs, 'r*')
#plt.show()
#plt.hold(True)
plt.subplot(2,1,2)
plt.plot(hyps, log_costs, 'b*')
plt.show()