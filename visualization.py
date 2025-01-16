import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt,patches
import random
import matplotlib.animation as anim



def save_anim(animation,filename):    
    FFwriter = anim.FFMpegWriter(fps=25)
    animation.save(filename, writer = FFwriter)
    



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


############################


    
from matplotlib.patches import Rectangle, FancyArrow

    
def flowplot(ax, loc, pcolors, v_0, grid_x, grid_y, fl_x, fl_y, title, overlay=None, grid_flow = False):
    arrow_scale = 30
    
    ax.set_title(title)
    
    # the particles
    sc = ax.scatter(loc[:,0], loc[:,1], marker='o', c=pcolors)
 
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
  
        i = 0
        for line in tracks_path:
            x = track[i][0][:idx]
            y = track[i][1][:idx]
            ax.plot(x, y, lw=1, color=pcolors[i])
            i+=1
            
        # remove and add arrrows at particles
        for a in arrows:
            a.remove()
        arrows.clear()
        
        if grid_flow:
            qv.set_UVC(*new_fl)
        
        i = 0            
        for item in new_loc:
            arrx = flow_speed_loc[i][0] * arrow_scale
            arry = flow_speed_loc[i][1] * arrow_scale
            arrow = FancyArrow(item[0],item[1],
                               arrx, arry, width = 0.04)
            a = ax.add_patch(arrow)
            arrows.append(a)
            i+=1

        if new_title:
            ax.set_title(new_title)
           
        return [sc,qv]
    
    # return update function
    return update

def animate(state, g_x, g_y, title, overlay=None, grid_flow = False):
    fig, ax = plt.subplots()
    x_0, track_0, idx_0, t_0,  v_0, fl_x, fl_y = state[0]
    #x_0, t_0, v_0 = state[0]
    
    
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(len(x_0))]

    
    update_plot = flowplot(ax, x_0, color, v_0, g_x, g_y,
                           fl_x,fl_y,
                           title.format(t_0),
                           overlay, grid_flow)

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
    


def random_v(mx,my):
    v = np.array([
                    (np.random.rand(my, mx) - 0.5) ,
                    (np.random.rand(my, mx) - 0.5)  
                    ])
    return v

def constant_v(mx,my):
    v = np.array([
                   np.array([0.2] * mx * my).reshape(my,mx)  ,
                     
                  np.array([-0.1] * mx * my).reshape(my,mx)
                  ])
    return v


def run(sigma, num = 20, do_anim = True):
    #T = 300 # time (300 hours)
    #NT = 100 # available time-points
    T=126
    NT=42
    
    TPIDX = T/NT # duration for each index in time
    N = 150 # step of simulation
    # num = 20 # number of particles (30)
    mx = NX + 1 # Grid size X , should be NX+1 
    my = NY + 1 # Grid size Y , should be XY+1
    epsilon = T/N # hours
    
    scale = mx # 3km per Gird unit
    
    #x_0 = scale*(np.random.rand(num, 2)*np.array([0.1, 0.5]) + np.array([0.1, 0.2]))
    #x_0 = np.random.rand(num, 2) * scale
    
    # x0_x = (np.random.rand(num) * NX).reshape(num,1)
    # x0_y = (np.random.rand(num) * NY).reshape(num,1)
    # x_0 = np.append(x0_x, x0_y, axis =1)
    
    
    #x_0 = np.array([[100,350]])
    
    print('num', num)
    # Gaussian
    mean=[100,350]    
    x_0 = np.random.normal(mean,sigma,(num,2))
    
    t = np.arange(0, T, epsilon) # Create the vector of times
    
    midpoint = lambda ar: (ar[1:] + ar[:-1])/2
    
    g_x_edges = np.linspace(0, mx, mx+1, endpoint=True) - 0.5
    g_y_edges = np.linspace(0, my, my+1, endpoint=True) - 0.5
    
    g_x, g_y = np.meshgrid(midpoint(g_x_edges), midpoint(g_y_edges))
    
    
    #####################################################
    
    v_field = constant_v(mx,my)
    #v_field = random_v(mx,my)
    
    UTest, VTest = v_field
    
        
    ###########################################
    
    # We will use this to extend the velocity field outside of the  range.
    clamp = lambda ar, min_, max_: np.maximum(min_, np.minimum(max_, ar))
    
    track_0 = []
    for i in range(num):
        track_0.append( [[],[]])
        
    state = [[x_0, track_0 ,0, 0, None, UTest, VTest]]
    
    # TODO add the time
    for i in range(N):

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
        
        UTest = U[:,:,time_idx]
        VTest = V[:,:,time_idx]
        #UTest, VTest = v_field
        
        vx = UTest[vi,vj]
        vy = VTest[vi,vj]
        
        #v = v_0_grid[:,vi-1,vj-1].T
        v = np.array([vx,vy]).T
        
        # print('v', v[0]) 
        # print('epsilon*v',epsilon*v)
        x = x + epsilon*v # Compute the next position value
        t = t + epsilon # Compute the next time
        
        # update calculated sspeed in previous record
        state[-1][4] = v
        
        state.append(
            [x, track, i , t, v, UTest, VTest]
            )
      
    
    #print (UTest[0,0], VTest[0,0])
    if do_anim :
        a = animate(state, g_x, g_y,
                    r"%d Particules Sigma %d Time= {:.2f}.h" % (num, sigma),
                    #overlay=plot_overlay, 
                    overlay=plot_overlay_p, 
                    grid_flow = False)
        
        save_anim(a,'sigma_%d.mp4' % (sigma))
        
    return state

# 1 particle no gaussian

# end 146.5557231, 451.544987 


#simgas = [1,2,3,5,6,6,7,8,9,10,15]


sigmas =  [0.5,1,2,3,4,5,6]
states = []
for  s in sigmas:
    print('sigma', s)
    state1 = run(sigma=s, num=100, do_anim=(False))
    states.append(state1)
    



# indices for N 150 and T 126
#  n int(H / e +1)
# 48h   n = 58
# 72   n = 86
# 120  n = 143
# N 

indic_time = { '48': 58,
              '72': 72,
              '120': 143
     }


def plot_states(sigma,time, do_plot=True,  do_initial=False):
    
    idx_sigm = sigmas.index(sigma)
    idx=indic_time[time]
    
    X0 = states[idx_sigm][0][0]
    x0 = X0[:,0] * 3
    y0 = X0[:,1] * 3    
    
    Xd = states[idx_sigm][idx][0]
    xd = Xd[:,0] * 3
    yd = Xd[:,1] * 3
    
    ux = np.average(xd) 
    uy = np.average(yd) 
    sx= np.std(xd) 
    sy = np.std(yd)
    
    mx = np.min(xd)
    my = np.min(yd)
    Mx = np.max(xd)
    My = np.max(yd)
    dx = Mx - mx
    dy = My - my
    print('s %2.1f t %3sh  xb %.1f yb %.1f sx %02.1f sy %02.1f mx %.1f Mx %.1f my %.1f My %.1f dx %.1f dy %.1f' % 
          (sigma*3,time,ux,uy,sx,sy,mx,Mx,my,My,dx,dy))
    
    if do_plot:
        fig, ax = plt.subplots()
        if do_initial:      
            plt.scatter(x0,y0,c='r',label='time 0')

        l = 5 ; b=l/2
        rectangle = patches.Rectangle((ux - b* sx, uy - b*sy), 
                                      l *sx, l *sy, edgecolor='orange',
                                      facecolor='none', label='%d sigma' % l)
        ax.add_patch(rectangle)
        cx = ux ; cy = uy + b * sy + 1
        ax.annotate('%.1f km' % (l*sx), (cx, cy), color='black', fontsize=10, 
                    ha='right', va='center')
        cx = ux - b * sx ; cy = uy
        ax.annotate('%.1f km' % (l*sy), (cx, cy), color='black', fontsize=10, 
                    ha='right', va='center')

        plt.scatter(xd,yd, label='time %sh' %time)
        plt.title('Sigma %.1fkm Time %sh, x_bar %.2f y_bar %.2f' % 
                  (sigma * 3,time,ux,uy))
        plt.legend(loc="lower right")
        plt.show()
    
for t in [ '48', '72', '120']:
    for s in sigmas:
    #for s in [1]:
        plot_states(s ,t, do_plot=False,do_initial=True)
        

# try 
# clean data
x = np.linspace(-10, 10, len(xd))
y = gauss1D(x, 20, 0 , 10)



# Executing curve_fit on noisy data
popt, pcov = curve_fit(gauss1D, x, xd)

ym = gauss1D(x, popt[0], popt[1], popt[2])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, ym, c='k', label='Model')
ax.scatter(x, xdd)

#popt, pcov = opt.curve_fit(twoD_Gaussian, (x,y), p0 = initial_guess)
