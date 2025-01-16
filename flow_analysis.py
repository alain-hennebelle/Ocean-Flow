import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt,patches
import random
import matplotlib.animation as anim
import numpy.linalg as lalg

mask_df = pd.read_csv('OceanFlow/mask.csv', header=None)
mask = mask_df.to_numpy()

time0u = pd.read_csv('OceanFlow/1u.csv', header=None)
time0v = pd.read_csv('OceanFlow/1v.csv', header=None)

# speed vector is a tuple
time0u_np = time0u.to_numpy()
time0v_np = time0v.to_numpy()

nx = time0u_np.shape[1]
ny =  time0u_np.shape[0]
NX=nx
NY=ny

# X 555
# Y 504



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

x  = [100,100,200]
y  = [450,250,350]

plt.scatter(100,350)
plt.scatter(x,y)
plt.imshow(masku, cmap='hot', interpolation='nearest',origin='lower')
plt.show()

# speed hot map time 0
plt.imshow(S[:,:,0], cmap='Blues', interpolation='nearest',origin='lower')
plt.scatter(170,121,marker='x')
plt.show()

# speed hotmap  time 99
plt.imshow(S[:,:,99], cmap='Blues', interpolation='nearest',origin='lower')
plt.scatter(170,121,marker='x')
plt.show()

plt.imshow(VAR, cmap='hot', interpolation='nearest',origin='lower')
plt.scatter(170,121,marker='x')
plt.show()


xd=  100
yd = 350

Ud = U[yd,xd]
Vd = V[yd,xd]


# each imte 72 hours, we need to extrpolate for  12 hours : 6 times more data
# maybe 6 hours 12 times more data

# Data every 3 days -> should be very 12 hours, 
# 𝐗1 : Multivariate Gaussian random variable. Flow data for no observation time
# times 
# 𝐗2 : Multivariate Gaussian random varibale. 
# Flow  data for the days we have observations. Time 0, 72h 
# 𝐱2 : Observations of  𝐗2 .
# 𝜇1 : Mean of  𝐗1 : Column 3 in the data, index 2, 3,
# 𝜇2 : Mean of  𝐗2 : Column 3 in the data index 1, 5, 9, ...

# number of desired indices ..100 * 6 so that time index is 12h

X2=Ud

# NTimes = 600
# data = np.zeros(NTimes)

data=X2
Ndata=X2.shape[0]
# ind_x2 = np.arange(0,NTimes,6)
# data[ind_x2] = X2
ind_x2 = np.arange(0,100,1)
perm = np.random.permutation(ind_x2.shape[0])


k1_x2, k2_x2 = [ ind_x2[a] for a in map(np.sort, np.split(perm, 2)) ]

for k_x2, label, c in [(k1_x2, "Partition 1", 'b'), (k2_x2, "Partition 2", 'r')]:
  plt.scatter(k_x2+1, data[k_x2], label=label, c=c, marker='x')

# x = np.arange(0,NTimes)
# plt.plot(x, data, c='k', label="Data")
# plt.legend()
# plt.show()

theta1 = 0.2   # a
theta2 = 3   # l
theta3 = 0.5 # modeling the noise tau

#  cross valiation
#covs = np.arange(0,NTimes)[:,None] - np.arange(0,NTimes)[None,:]
covs = np.arange(0,Ndata)[:,None] - np.arange(0,Ndata)[None,:]

covs  = covs * 72
sigma = theta1 * np.exp( - covs**2 / theta2**2 )

sigma_11 = sigma[k1_x2[:,None],k1_x2]
sigma_12 = sigma[k1_x2[:,None],k2_x2]
sigma_21 = sigma[k2_x2[:,None],k1_x2]
sigma_22 = sigma[k2_x2[:,None],k2_x2]

#mu_1 = data[k1_x2]
#mu_2 = data[k2_x2]
# mylist = data[k1_x2]

def runningMean(x, N):
    y = np.zeros((len(x),))
    for ctr in range(len(x)):
         y[ctr] = np.sum(x[ctr:(ctr+N)])
    return y/N

mu_1 = runningMean(data[k1_x2],10)
mu_2 = runningMean(data[k2_x2],10)




sigma_22_noise = sigma_22 + theta3 * np.eye(mu_2.shape[0])
x2_2 = data[k2_x2]

mu_1_2 = mu_1 + sigma_12.dot( lalg.solve(sigma_22_noise, x2_2-mu_2 ) )
mu_2_2 = mu_2 + sigma_22.dot( lalg.solve(sigma_22_noise, x2_2-mu_2 ) )

sigma_1_2 = sigma_11 - sigma_12.dot(lalg.inv(sigma_22_noise).dot(sigma_12.T))
sigma_2_2 = sigma_22 - sigma_22.dot(lalg.inv(sigma_22_noise).dot(sigma_22.T))


# ind_x1 = np.array(sorted(list(set(np.arange(0, 360)) - set(ind_x2))))
# We have computed the conditional distributions of the Partition 1 given Partition 2. Let's see how this looks like
plt.plot(k1_x2+1, mu_1_2, c='b', label="Conditional Mean P1 given P2")
plt.scatter(k1_x2+1, data[k1_x2], c='orange', marker='o', label="P1 Measured Data")

plt.xlabel("Time");
plt.ylabel("Flow in X");

sigma_diags = np.diagonal(sigma_1_2)
plt.plot(k1_x2+1, mu_1_2 + 2*np.sqrt(sigma_diags), c='r', label="2 Sigmas above the mean")
plt.plot(k1_x2+1, mu_1_2 - 2*np.sqrt(sigma_diags), c='k', label="2 Sigmas below the mean")

plt.legend()
plt.show()




# A Plot of the Conditional Distributions
nu_mu = np.zeros(100)
nu_mu[k1_x2] = mu_1_2
nu_mu[k2_x2] = mu_2_2

new_sigma_diags = np.zeros(100)
new_sigma_diags[k1_x2] = sigma_diags
new_sigma_diags[k1_x2] = np.diagonal(sigma_2_2)

#plt.plot(ind_x2, (nu_mu + 2*np.sqrt(new_sigma_diags))[::3], c='r')
#plt.plot(ind_x2, (nu_mu - 2*np.sqrt(new_sigma_diags))[::3], c='r')
plt.plot(ind_x2, nu_mu, c='k')
plt.plot(ind_x2, (nu_mu + 2*np.sqrt(new_sigma_diags)), c='r')
plt.plot(ind_x2, (nu_mu - 2*np.sqrt(new_sigma_diags)), c='r')
plt.show()




###############################




####################################################%%%%%%%%%%%%%%%%%
###################################

    
import scipy.stats
import tqdm


covs = np.arange(0,Ndata)[:,None] - np.arange(0,Ndata)[None,:]
covs  = covs * 72


def do_it(xd,yd,  Array, a_vals, ell_vals, tau = 0.001, K=10):
    Vd = Array[yd,xd]
    X2= Vd

    #a_vals = np.arange(a_vals_range)
    #ell_vals = np.arange(7.2, 360, 2)
    
    #ind_x2 = np.arange(0, 66, 4)
    #ind_x2 = np.arange(0,NTimes,6)
    
    
    # NTimes = 600
    # data = np.zeros(NTimes)
    
    data=X2
    
    Ndata=X2.shape[0]
    
    perm = np.random.permutation(ind_x2.shape[0])
    ks_x2 = [ ind_x2[a] for a in map(np.sort, np.array_split(perm, K)) ]
    
    
    target_best = -np.inf
    params_best = {}
    
    i = 0
    j = 0
    perfs = np.zeros((len(a_vals),len(ell_vals)))
    for a_param in tqdm.tqdm(a_vals):
    
        j =0
        for ell_param in ell_vals:
            sigma = a_param * np.exp( - covs**2 / ell_param**2 )
        
            vals = []
          
            for k in range(K):
                x_test = ks_x2[k]
                x_train = np.sort(np.array(list(set(ind_x2) - set(x_test))))
            
    
                mu_1 = runningMean(data[x_test],5)
                mu_2 = runningMean(data[x_train],5)
            
                sigma_11 = sigma[x_test[:,None],x_test]
                sigma_12 = sigma[x_test[:,None],x_train]
                sigma_21 = sigma[x_train[:,None],x_test]
                sigma_22 = sigma[x_train[:,None],x_train]
                
                sigma_22_noise = sigma_22 + tau * np.eye(mu_2.shape[0])
                x2_2 = data[x_train]
                
                mu_1_2 = mu_1 + sigma_12.dot( lalg.solve(sigma_22_noise, x2_2 - mu_2 ) )
                sigma_1_2 = sigma_11 - sigma_12.dot(lalg.inv(sigma_22_noise).dot(sigma_12.T))
                
                val = scipy.stats.multivariate_normal.logpdf(data[x_test], mean=mu_1_2, cov=sigma_1_2)
                vals.append(val)
                
            total = sum(vals)
            perfs[i,j] = total
    
            if total > target_best:
                params_best = dict(
                    a=a_param,
                    ell=ell_param,
                    xs=x_test,
                    data=data[x_test],
                    mu=mu_1_2,
                    sigma=np.sqrt(np.diagonal(sigma_1_2)),
                    perf = total
                )
                target_best = total
            j = j+1    
        i = i+1  
    
    sigma_opt = params_best['a'] * np.exp( - covs**2 / params_best['ell']**2 )
    plt.scatter(params_best['xs'], params_best['data'], c='b', marker='o', label="Data")
    plt.scatter(params_best['xs'], params_best['mu'], c='orange', marker='x', label=r"Predicted $\mu$")
    plt.plot(params_best['xs'], params_best['mu'] + 2*params_best['sigma'], c='r', label="2 sigma band")
    plt.plot(params_best['xs'], params_best['mu'] - 2*params_best['sigma'], c='r')
    plt.title(r"Best parameters: $a={a}$, $l={ell}$".format(**params_best))
    plt.legend()
    plt.show()
    
    print('a', params_best['a'], ',l',params_best['ell'], ',perf', params_best['perf'])

    return perfs


a_s=np.arange(0.001, 0.05, 0.001)
l_s=np.arange(7.2,360,3)


pv=do_it(100,350,V, a_s, l_s,K=10)



plt.imshow(p, cmap='hot', interpolation='nearest')
plt.show()

#ax = sns.heatmap(p, linewidth=0.5)
ax = sns.heatmap(p)
#ax.axis([0.001,0.05, 7.2,360])
plt.show()


ax = sns.heatmap(np.log(p))
plt.xlabel('l param values')
plt.ylabel('a paran values')
plt.show()

y, x = np.meshgrid(l_s, a_s)
fig, ax = plt.subplots()
c = ax.pcolormesh(x, y, p, cmap='RdBu')
ax.set_title('Performance / kernel parameters')
plt.ylabel('l param values')
plt.xlabel('a paran values')
# set the limits of the plot to the limits of the data
#ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)


(ai, li) = np.unravel_index(np.argmax(p),p.shape)
a_s[ai]
l_s[li]
print('Optimum  a: %.3f  l: %.3f' % (a_s[ai],l_s[li]))

pu=do_it(100,350,U, a_s, l_s, tau=0.1,K=10)
pv=do_it(100,350,V, a_s, l_s, tau=0.1,K=10)

pu=do_it(100,450,U, a_s, l_s,K=10)
pv=do_it(100,450,V, a_s, l_s,K=10)

pu=do_it(100,250,U, a_s, l_s,K=10)
pv=do_it(100,250,V, a_s, l_s,K=10)

pu=do_it(200,350,U, a_s, l_s,K=10)
pv=do_it(200,350,V, a_s, l_s,K=10)

#  t=0.1    perf U 68.36 perf V 87.9
# t=0.01    perf U 102.1 perf V
# t=0.001  100.58
# t=0.0001  

#############################
ts =[0.1,0.01,0.001,0.0001]
ps = [ 68.35, 81.8, 100.58, 102.1]

plt.plot(np.log(ts),ps)
#plt.plot(ts,ps)
plt.title('Performance vs Tau value')
plt.xlabel('log(Tau)')
plt.ylabel('Performance')
plt.show()
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

# Problem 5




def interpolate(NTimes, X2, Tau_opt, a_opt, l_opt, axis_name,do_plot):

    ind_x = np.arange(0, NTimes)
    ind_x2 = np.arange(0, NTimes, NTimes//100)
    
    ind_x1 = np.array(sorted(list(set(ind_x) - set(ind_x2))))
    
    data=np.zeros(NTimes)
    data[ind_x2] = X2
    
    covs = np.arange(0,NTimes)[:,None] - np.arange(0,NTimes)[None,:]
     
    sigma_opt = a_opt * np.exp( - covs**2 / l_opt_u**2 )
    
    sigma_11_n = sigma_opt[ind_x1[:,None],ind_x1];
    sigma_12_n = sigma_opt[ind_x1[:,None],ind_x2];
    sigma_21_n = sigma_opt[ind_x2[:,None],ind_x1];
    sigma_22_n = sigma_opt[ind_x2[:,None],ind_x2];
    
    #mu_x1 = runningMean(data[ind_x1],5)
    #mu_x2 = runningMean(data[ind_x2],5)
    mu_x1 = np.zeros(ind_x1.shape[0])
    mu_x2 = np.zeros(ind_x2.shape[0])
    
    sigma_22_n_noise = sigma_22_n + Tau_opt * np.eye(mu_x2.shape[0])
    x2 = data[ind_x2]
    
    mu_x1_n = mu_x1 + sigma_12_n.dot( lalg.solve(sigma_22_n_noise, x2 - mu_x2 ) )
    mu_x2_n = mu_x2 + sigma_22_n.dot( lalg.solve(sigma_22_n_noise, x2 - mu_x2 ) )
    
    sigma_x1_n = sigma_11_n - sigma_12_n.dot(lalg.inv(sigma_22_n_noise).dot(sigma_12_n.T))
    sigma_x2_n = sigma_22_n - sigma_22_n.dot(lalg.inv(sigma_22_n_noise).dot(sigma_22_n.T))
    
    new_mu = np.zeros(NTimes)
    new_mu[ind_x1] = mu_x1_n
    new_mu[ind_x2] = mu_x2_n
    
    new_sigma = np.zeros(NTimes)
    new_sigma[ind_x1] = np.sqrt(np.diagonal(sigma_x1_n))
    new_sigma[ind_x2] = np.sqrt(np.diagonal(sigma_x2_n))
    
    data[ind_x1] = new_mu[ind_x1]
    
    if do_plot:
        xs = np.arange(0, NTimes)
        plt.figure(figsize=(12,8))
        plt.plot(xs, new_mu, c='r', label=r"$\mu_{X_1 | X_2}$", lw=0.5)
        plt.plot(xs, new_mu + 3*new_sigma, c='k', label="3 Sigma Upper Bound", lw=0.5)
        plt.plot(xs, new_mu - 3*new_sigma, c='b', label="3 Sigma Lower Bound", lw=0.5)
        plt.scatter(ind_x2, data[ind_x2], c='r', marker='o', label="Observed data ")
        #plt.scatter(ind_x1, data[ind_x1], c='y', marker='X', label="Interpolated data")
        plt.title('Interpolation of data every 12h %s' % axis_name),
        plt.xlabel("Time")
        plt.ylabel("Velocity")
        plt.legend()
        plt.show()
    
    return data


NTimes = 300

xd=  100
yd = 350

Ud = U[yd,xd]
Vd = V[yd,xd]

X2=Ud
Tau_opt =  0.001
a_opt_u = 0.015
l_opt_u = 85.2
a_opt_v = 0.022
l_opt_v = 88.2

# interpolate(NTimes,Ud, Tau_opt, a_opt_u, l_opt_u,'velocity along X',do_plot=False)

# interpolate(NTimes,Vd, Tau_opt, a_opt_v, l_opt_v,'velocity along Y')

# new velocity fields

UU = np.zeros( (NY,NX,NTimes))
VV = np.zeros((NY,NX,NTimes))


for xd in range (0,NX):
    print('%d / %d' % (xd, NX))
    for yd in range(0,NY):
        
        Ud = U[yd,xd]
        Vd = V[yd,xd]
        # vectors over 600 time points    
        NewU = interpolate(NTimes,Ud, Tau_opt, a_opt_u, l_opt_u,
                           'velocity along X', do_plot = False)
        NewV = interpolate(NTimes, Vd, Tau_opt, a_opt_v, l_opt_v,
                           'velocity along Y', do_plot = False)
        UU[yd,xd] = np.copy(NewU)
        VV[yd,xd] = np.copy(NewV)

    
from matplotlib.patches import Rectangle, FancyArrow

    
def flowplot(ax, loc, pcolors, v_0, grid_x, grid_y, fl_x, fl_y, title, overlay=None, 
             grid_flow = False, track_arrow = True):
    arrow_scale = 30
    
    ax.set_title(title)
    
    # the particles
    if track_arrow:
        sc = ax.scatter(loc[:,0], loc[:,1], marker='o', s=4, c=pcolors)
    else:
        sc = ax.scatter(loc[:,0], loc[:,1], marker='o', s=10, c='r')
        
    tracks_path = []
    for item in loc:
        line = ax.plot([], [], lw=1)
        tracks_path.append(line)
    
    #if flow_speed:
        #qv = ax.quiver(loc[:,0], loc[:,1], flow_speed[:,0], flow_speed[:,1])    
    qv = None
    if (grid_flow):
        qv = ax.quiver(grid_x, grid_y, fl_x, fl_y)        
    # add and arrow for flow speed at particle place
    #print(flow_speed)
    i = 0
    arrows = []
    for item in loc:
        arrow = FancyArrow(item[0],item[1],v_0[i][0],v_0[i][1], 
                 width = 0.04)
        a = ax.add_patch(arrow)
        arrows.append(a)
        i+=1

    if overlay:
      overlay(ax)
      
    def update(new_loc, track, idx, t, flow_speed_loc, new_fl = None, new_title=None):      
        # update particles location
        sc.set_offsets(new_loc)
 
        # redraw the tracks
  
        if track_arrow:
            i = 0
            for line in tracks_path:
                x = track[i][0][:idx]
                y = track[i][1][:idx]
                ax.plot(x, y, lw=0.5, color=pcolors[i])
                i+=1
                
            # remove and add arrrows at particles
            for a in arrows:
                a.remove()
            arrows.clear()
        
        if grid_flow:
            qv.set_UVC(*new_fl)
        
        i = 0     
        stopped_part = []
        zero = 1e-3
        for item in new_loc:                
            if np.abs(flow_speed_loc[i][0]) > zero or  (flow_speed_loc[i][1]) > zero:
                if track_arrow:
                    arrx = flow_speed_loc[i][0] * arrow_scale
                    arry = flow_speed_loc[i][1] * arrow_scale
                    arrow = FancyArrow(item[0],item[1],
                                    arrx, arry, width = 0.04)
                    a = ax.add_patch(arrow)
                    arrows.append(a)
                i+=1
            else:
                stopped_part.append(i)
        
        if len(stopped_part) != 0:
            ax.scatter(new_loc[stopped_part][:,0],
                        new_loc[stopped_part][:,1],
                        marker='o',c='red')
                
        if new_title:
            ax.set_title(new_title)
           
        return [sc,qv]
    
    # return update function
    return update

def animate(state, g_x, g_y, title, overlay=None, grid_flow = False, track_arrow=True):
    fig, ax = plt.subplots()
    x_0, track_0, idx_0, t_0,  v_0, fl_x, fl_y = state[0]
    #x_0, t_0, v_0 = state[0]
    
    
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(len(x_0))]

    
    update_plot = flowplot(ax, x_0, color, v_0, g_x, g_y,
                           fl_x,fl_y,
                           title.format(t_0),
                           overlay, grid_flow,track_arrow)

    def frame(s):
        x, track,idx, t, v,fl_x, fl_y  = s
        return update_plot(x, track, idx, t, v,
                        new_fl = (fl_x, fl_y),
                        new_title=title.format(t))

    animation = anim.FuncAnimation(fig, frame, frames=state, blit=False)
    plt.close()
    return animation



def plot_overlay(ax):
    import matplotlib.patches
    #ax.scatter(g_x, g_y, marker='o', c='r')
    # ax.add_patch(matplotlib.patches.Rectangle((0, 0), 1000, 100, edgecolor='k', facecolor='orange'))
    # ax.text(scale*0.5,scale*0.05,'Land')
    # ax.text(scale*0.5,scale*0.5,'Sea')
    ax.axis([-0.5, np.max(g_x) + 0.5, -0.5, np.max(g_y) + 0.5])
  
def plot_overlay_p(ax):
    ax.imshow(masku, cmap='hot', interpolation='nearest',origin='lower')
    ax.axis([0, NY, 0, NX])  
    
def save_anim(animation,filename):    
    FFwriter = anim.FFMpegWriter(fps=25)
    animation.save(filename, writer = FFwriter)




def run(sigm, UU, VV, filename,num = 20, Niter=200, do_anim = True,
        track_arrow = True):
    #T = 300 # time (300 hours)
    #NT = 100 # available time-points
    T=300*24  # in hours
    NT=UU.shape[2]   # available time-points  24h each
    
    TPIDX = T/NT # duration for each index in time
    #N = 200 # step of simulation
    # num = 20 # number of particles (30)
    mx = NX + 1 # Grid size X , should be NX+1 
    my = NY + 1 # Grid size Y , should be XY+1
    epsilon = T/Niter # hours
    

    print('num', num)
    # Gaussian
    mean=[100,350]    
    
    # x_0 = np.random.normal(mean,sigm,(num,2))
    # # for printing
    # sigm  = int(sigm*3)
    
    c = 0
    xs = []
    for i in range(num **2):
        x = np.random.randint(NX)
        y = np.random.randint(NY)
        if masku[y,x] == 1:
            xs.append([x,y])
            c=c+1
        if c == num:
            break    
    x_0 = np.array(xs)
        
    
    t = np.arange(0, T, epsilon) # Create the vector of times
    
    midpoint = lambda ar: (ar[1:] + ar[:-1])/2
    
    g_x_edges = np.linspace(0, mx, mx+1, endpoint=True) - 0.5
    g_y_edges = np.linspace(0, my, my+1, endpoint=True) - 0.5
    
    g_x, g_y = np.meshgrid(midpoint(g_x_edges), midpoint(g_y_edges))
    
  
    # We will use this to extend the velocity field outside of the  range.
    clamp = lambda ar, min_, max_: np.maximum(min_, np.minimum(max_, ar))
    
    track_0 = []
    for i in range(num):
        track_0.append( [[],[]])

    UTest = UU[:,:,0]
    VTest = VV[:,:,0]
    state = [[x_0, track_0 ,0, 0, None, UTest, VTest]]
    
    # TODO add the time
    for i in range(Niter):

        x, track, idx, t, v, _, _  = state[-1]
    
        j = 0
        for item in x:
            track[j][0].append(item[0])
            track[j][1].append(item[1])    
            j=j+1
    
        vi = clamp(np.searchsorted(g_x_edges, x[:,1]), 1, my-1) -1
        vj = clamp(np.searchsorted(g_y_edges, x[:,0]), 1, mx-1) -1
        
        # flow at time idx
        time_idx = int(t/TPIDX)
        
        UTest = UU[:,:,time_idx]
        VTest = VV[:,:,time_idx]
        #UTest, VTest = v_field
        
        vx = UTest[vi,vj]
        vy = VTest[vi,vj]
        
        #v = v_0_grid[:,vi-1,vj-1].T
        v = np.array([vx,vy]).T
        # speed in km per hours , not grid unit
        vr = v/3
        
        # print('v', v[0]) 
        # print('epsilon*v',epsilon*v)
        x = x + epsilon*vr # Compute the next position value
        t = t + epsilon # Compute the next time
        
        # update calculated sspeed in previous record
        state[-1][4] = v
        
        state.append(
            [x, track, i , t, v, UTest, VTest]
            )
      
    #print (UTest[0,0], VTest[0,0])
    if do_anim :
        a = animate(state, g_x, g_y,
                    r"%d Particules Time= {:.2f}.h" % (num),
                    #overlay=plot_overlay, 
                    overlay=plot_overlay_p, 
                    grid_flow = False, track_arrow = track_arrow)
        
        save_anim(a,filename +'.mp4' )
        
    return state




#sigmas =  [0.5,1,2,3,4,5,6]

states = []
state1 = run(4, U,V,num=20, Niter=300, do_anim=True)

state1 = run(4, UU,VV,num=30, Niter=300, do_anim=True)

#state1 = run(3, UU,VV,num=10, Niter=200, do_anim=True)
states.append(state1)
    
