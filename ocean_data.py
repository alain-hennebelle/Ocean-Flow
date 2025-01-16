import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt  #



mask_df = pd.read_csv('OceanFlow/mask.csv', header=None)
mask = mask_df.to_numpy()

time0u = pd.read_csv('OceanFlow/1u.csv', header=None)
time0v = pd.read_csv('OceanFlow/1v.csv', header=None)

# speed vector is a tuple
time0u_np = time0u.to_numpy()
time0v_np = time0v.to_numpy()

nx = time0u_np.shape[1]
ny =  time0u_np.shape[0]

# X 555
# Y 504

vector0 = np.ndarray((ny,nx),dtype=tuple)
for i in range(ny):
    for j in range(nx):
        vector0[i,j] = (time0u_np[i,j],time0v_np[i,j])

speed0 = np.sqrt(time0u_np**2 + time0v_np**2)


U = np.zeros((ny,nx,100))
V = np.zeros((ny,nx,100))
for i in range(100):
    u = pd.read_csv('OceanFlow/%du.csv' % (i+1), header=None).to_numpy()
    U[:,:,i] = u
    v = pd.read_csv('OceanFlow/%dv.csv' % (i+1), header=None).to_numpy()
    V[:,:,i] = v    
    
# speeds   
S =  np.sqrt(U**2 + V**2)

# variance in time
VAR=np.var(S,axis= 2)

x_min = 0
y_min = 0
minim = np.Infinity

for i in range(ny):
    for j in range(nx):
        if VAR[i,j] != 0 and  VAR[i,j] < minim:
            x_min = j
            y_min = i
            minim = VAR[i,j]

print ('minimum variance %g  at x = %d, y = %d' % (minim, x_min,y_min) )

# map of the philippines
masku = np.flipud(mask)

plt.imshow(masku, cmap='hot', interpolation='nearest',origin='lower')
plt.show()

plt.imshow(S[:,:,0], cmap='Blues', interpolation='nearest',origin='lower')
plt.scatter(170,121,marker='x')
plt.show()

plt.imshow(S[:,:,99], cmap='Blues', interpolation='nearest',origin='lower')
plt.scatter(170,121,marker='x')
plt.show()

plt.imshow(VAR, cmap='hot', interpolation='nearest',origin='lower')
plt.scatter(170,121,marker='x')
plt.show()

# problem 1.b
np.max(U)
np.unravel_index(np.argmax(U, axis=None), U.shape)
#(181, 347, 28)

U[181,347,28]






Avg_U = np.average(U)
Avg_V = np.average(V)


import scipy.signal as signal
def xcorr(x,y):
    """
    Perform Cross-Correlation on x and y
    x    : 1st signal
    y    : 2nd signal

    returns
    lags : lags of correlation
    corr : coefficients of correlation
    """
    corr = signal.correlate(x, y, mode="full")
    #lags = signal.correlation_lags(len(x), len(y), mode="full")
    #return lags, corr
    return corr

np.corrcoef(U[10,10,:],U[100,100:])


def distance(i,j,k,l):    
    return np.sqrt((k-i)**2+(l-j)**2)



# for i in range(ny):
#     for j in range(nx):
#         for k in range(ny):
#             for l in range(nx):
       
sample_size = 100         
index_i = random.sample(range(ny),k=sample_size) 
index_j = random.sample(range(nx),k=sample_size) 
index_k = random.sample(range(ny),k=sample_size) 
index_l = random.sample(range(nx),k=sample_size) 

max_c1 = -np.Infinity
max_c2 = -np.Infinity

threshold = 0.98
result1 = []
result2 = []
for i in index_i:
    for j in index_j:
        for k in index_k:
            for l in index_l:
                #if distance(i,j,k,l) > 5 and np.max(U[i,j:]) > 0 and np.max(U[k,l:]) > 0:
                d = distance(i,j,k,l)
                if d > 20:                    
                    #print(' calculate ',i,j,k,l)
                    c1=np.corrcoef(U[i,j,:],U[k,l,:])[0,1]
                    c2=np.corrcoef(V[i,j,:],V[k,l,:])[0,1]
                    if c1 > threshold:
                        print('C1 (%d,%d) and (%d,%d) d=%2f c1 %f, c2 %f' % (j,i,l,k, d,c1,c2))
                        result1.append( [(i,j,k,l,d),c1,c2])
                    if c2 > threshold: 
                        result2.append( [(i,j,k,l,d),c2,c1])
                    if (c1 > max_c1):
                        max_c1 = c1
                        max_c1_tuple = (i,j,k,l)
                    if (c2 > max_c2):
                        max_c2 = c2
                        max_c2_tuple = (i,j,k,l)       

                        
# C2 (110,449) and (117,18) d=431 c2 0.985191, c1 0.783047                
# C2 (416,466) and (441,447) d=31 c2 0.985750, c1 0.875231

sample_size = 80         
index_i = random.sample(range(ny),k=sample_size) 
index_j = random.sample(range(nx),k=sample_size) 
index_k = random.sample(range(ny),k=sample_size) 
index_l = random.sample(range(nx),k=sample_size) 

result3 = []
threshold = 0.96
for i in index_i:
    for j in index_j:
        for k in index_k:
            for l in index_l:
                #if distance(i,j,k,l) > 5 and np.max(U[i,j:]) > 0 and np.max(U[k,l:]) > 0:
                d = distance(i,j,k,l)
                if d > 33:                    
                    #print(' calculate ',i,j,k,l)
                    c1=np.corrcoef(U[i,j,:],U[k,l,:])[0,1]
                    c2=np.corrcoef(V[i,j,:],V[k,l,:])[0,1]
                    a=(c1+c2) / 2
                    if a > threshold:
                        print('(%d,%d) and (%d,%d) d=%2f c1 %f, c2 %f a %f' % (j,i,l,k, d,c1,c2,a))
                        result3.append( [i,j,k,l,d,c2,c1,a])
                        
# (494,134) and (505,172) d=39.560081 c1 0.959951, c2 0.966139
# (494,134) and (505,169) d=36.687873 c1 0.970107, c2 0.968548        
# (334,257) and (337,174) d=83.054199 c1 0.978903, c2 0.957421

# (334,257) and (341,177) d=80.305666 c1 0.979934, c2 0.967465

# (334,257) and (344,181) d=76.655072 c1 0.985182, c2 0.959892

#array([178.        , 341.        , 264.        , 337.        ,
#        86.092973  ,   0.98073731,   0.99126111,   0.98599921])

# map of the philippines
# 
np.corrcoef(U[178,341,:],U[264,337,:])[0,1]
np.corrcoef(V[178,341,:],V[264,337,:])[0,1]
U1=U[178,341,:]
V1=V[178,341,:]

U2=U[264,337,:]
V2=V[264,337,:]

np.corrcoef(U1,U2)
np.corrcoef(V1,V2)


plt.imshow(mask, cmap='hot', interpolation='nearest')
plt.scatter(341,ny - 178,marker='o',color='r',s=80)
plt.scatter(337,ny - 264,marker='o',color='r',s=80)


plt.imshow(masku, cmap='hot', interpolation='nearest',origin='lower')
plt.x(3)
plt.yscale(3)
plt.show()

def plot_correlated(a,c):
    U1=U[a[0],a[1],:]
    V1=V[a[0],a[1],:]

    U2=U[a[2],a[3],:]
    V2=V[a[2],a[3],:]

    cc1 =np.corrcoef(U1,U2)[0,1]
    cc2 = np.corrcoef(V1,V2)[0,1]
    print('%05f %05f %05f' % (cc1,cc2,(cc1+cc2)/2))
    
    plt.imshow(masku, cmap='hot', interpolation='nearest',origin='lower')
    plt.xlabel('distance in grid units')
    plt.ylabel('distance in grid units')
    plt.title('Long range correlations')
    plt.scatter(a[1], a[0],marker='o',color=c,s=150)
    plt.scatter(a[3], a[2],marker='o',color=c,s=150)
    plt.show()
    
# (39,121) and (147,118) 
c1 = [121,38,118,147]
c2 = [121,39,118,147]
c3 = [121,38,118,148]
c10 = [178,341,264,337]


# plot_correlated(341,178,337,264, 'r')
# plot_correlated(121,39,118,147, 'b')

plot_correlated(c1, 'b')
plot_correlated(c3, 'g')
plot_correlated(c2, 'r')
# zoom
