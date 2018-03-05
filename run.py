import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import BScode
import LFcode
import animate
import plot

#####
#
#   The Initial Conditions
#
#####

#General
alpha = 0.02
L = 20
h_div = 200
dx = L/h_div
dt = dx/10

#Lax-Friedrichs Model
center = 5
kpeak = 1.35#3.5
eta0 = [np.exp(-.5*(x-center)**2)+1 for x in np.arange(0,L,dx)]
k0 = [kpeak*np.exp(-.5*(x-center)**2)+1 for x in np.arange(0,L,dx)]

#Verlet
kmin = 2.4 #2.5
kmax = 100 # 10 (doesn't mean anything for getk_d)
dk = kmin*1
g = 0
m = 1
gamma = 0.1 #0.1
a0 = (L/h_div)/5

LFsteps = 100
BSsteps = 300 #1000


#####
#
#   The Code
#
#####

eta,xyeta,v,xyv = LFcode.doLF(alpha,L,dx,dt,LFsteps,eta0,k0,True)

in_x0 = xyeta[LFsteps-1]
in_xm1 = xyeta[LFsteps-2]

in_v0 = v[LFsteps-1]
x0 = BScode.data_to_pairs(in_x0)
xm1 = BScode.data_to_pairs(in_xm1)
v0 = np.array([[v,0] for v in in_v0])



BSdt= dt*2

spring_data,energy = BScode.verlet(x0,xm1,v0,BScode.acceleration,a0,m,kmin,kmax,dk,alpha,gamma,dt,BSdt,BSsteps,False)

#Format the verlet data to the lax-friedrichs format
data_fmt = []
for t in spring_data:
    xout = []
    yout = []
    for d in t:
        xout.append(d[0])
        yout.append(d[1])
    data_fmt.append([xout,yout])

"""

x = [2.53240, 1.91110, 1.18430, 0.95784, 0.33158,
     -0.19506, -0.82144, -1.64770, -1.87450, -2.2010]

y = [-2.50400, -1.62600, -1.17600, -0.87400, -0.64900,
     -0.477000, -0.33400, -0.20600, -0.10100, -0.00600]

coefficients = polyfit(x, y, 6)
polynomial = poly1d(coefficients)
xs = arange(-2.2, 2.6, 0.1)
ys = polynomial(xs)

plot(x, y, 'o')
plot(xs, ys)
ylabel('y')
xlabel('x')
show()
"""

#combine the data
full_data = []
for time in xyeta:
    x = []
    for pos in time[0]:
        x.append(pos)
    y = []
    for pos in time[1]:
        y.append(pos)
    full_data.append([x,y])

for time in data_fmt:
    x = []
    for pos in time[0]:
        x.append(pos)
    y = []
    for pos in time[1]:
        y.append(pos)
    full_data.append([x,y])

"""
for t in full_data:
    y = t[1]
    print(max(y))
"""
break_dat = full_data[250]
peak_h = max(break_dat[1])
i = 0
while i < len(break_dat[1]):
    if break_dat[1][i] == peak_h:
        print(i)
        peak_i = i
    i += 1
print(break_dat[0][peak_i],break_dat[1][peak_i])

#####
#
#   The Animation
#
#####
video_length = 7
animate.animate(alpha,L,dx,dt,full_data,video_length,True)

show_frame = False

#print(np.array(data))

if show_frame:
    # Max of 4 frames (for more add colors and linestyles)
    print("Loading Image...")
    frames = [250,275,300]
    #xvals = [i for [i,j] in indata]
    #yvals = [j for [i,j] in indata]
    #scatter(xvals,yvals,'$x$','$h$',None,False)
    i = 0
    out = []
    while i < len(frames):
        index = frames[i]
        d = full_data[index]
        xs = []
        ys = []
        out.append(d)
        """
        for [x,y] in d:
            xs.append(x)
            ys.append(y)
        out.append(np.array([np.array(xs),np.array(ys)]))
        """
        i += 1
    plot.plot_frame(out,None,alpha,L,xlabel='$q$',ylabel='$\eta$',title='')