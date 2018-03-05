import numpy as np
from math import sqrt
import numpy as np
from scipy.signal import argrelextrema


def acceleration(xn,vn,a0,m,k,gamma):
	"""
	This function calculates the acceleration of all points
	at a given time
	"""
    aout = np.zeros((len(xn),2))
    j = 0

    rp = xn[j+1]-xn[j]

    potl_energy = [0.5*k[j]*np.linalg.norm(rp)**2]
    while j <= len(xn)-1:
        am = np.zeros(2)
        ap = np.zeros(2)

        if j == 0:
            
            rp = xn[j+1]-xn[j]
            
            ap.fill(0)

            potl_energy.append(0.5*k[j]*np.linalg.norm(rp)**2)

        if 0 < j < len(xn)-1:
            rp = xn[j+1]-xn[j]
            rphat = np.array(rp/sqrt(np.dot(rp,rp)))
            ap = (k[j]/m) * (sqrt(np.dot(rp,rp))- a0) * rphat

            rm = np.array(xn[j-1]-xn[j])
            rmhat = np.array(rm/sqrt(np.dot(rm,rm)))
            am = np.array((k[j]/m) * (sqrt(np.dot(rm,rm))-a0)*rmhat)

            potl_energy.append(0.5*k[j]*np.linalg.norm(rp)**2+0.5*k[j]*np.linalg.norm(rm)**2)

        if j == len(xn):
            
            rm = np.array(xn[j-1]-xn[j])
           
            am.fill(0)
            
            potl_energy.append(0.5*k[j]*np.linalg.norm(rm)**2)

        #To add gravity:
        #afixed = np.array([0,-m*g])

        afixed = np.array([0,0])
        aout[j] = ap + am + afixed -gamma*vn[j]

        j += 1

    return aout,potl_energy

def getk_th(xn,kmin,kmax):
	"""
	This function adjusts all the spring constants based on
	the angles between nearby points
	(Not used in final version)
	"""
    k = np.zeros(len(xn))
    k.fill(kmin)
    for i in range(1,len(xn)-1):
        ra = np.array(xn[i-1]-xn[i])
        rb = np.array(xn[i+1]-xn[i])

        rahat = ra/sqrt(np.dot(ra,ra))
        rbhat = rb/sqrt(np.dot(rb,rb))

        vec = np.dot(rahat,rbhat)

        if np.abs(vec) > 1:
            vec = 1


        th = np.abs(np.arccos(vec)%np.pi) #angle of two springs with ball i
        
        k[i] = kmax*(1-th/np.pi)+kmin

    return k

def getk_d(xn,kmin,kmax,alpha,dk):
	"""
	This function calculates the spring constants based on height
	and depth of the water at the x-y position of the point
	"""
    k = np.zeros(len(xn))
    factor = np.zeros(len(xn))
    k.fill(kmin)

    amb_height = xn[0][1]
    j = 0
    while j < len(xn)-1:
        bottom_height = xn[j][0]*alpha
        depth = amb_height - bottom_height
        wave_height = xn[j][1] - amb_height
        factor[j] = np.abs(wave_height/depth)
        k[j] = kmin + factor[j]*dk

        j += 1
    #print(factor)
    return k

def getk_e(xn,kmin,kmax,a,alpha,const):
	"""
	This function adjusts the spring constant based on what the energy
	should be for the point
	"""
    k = np.zeros(len(xn))
    k.fill(kmin)
    amb_height = xn[0][1]
    j = 1
    while j < len(xn)-1:
        strech = np.linalg.norm(np.abs(xn[j]-xn[j-1])-a)
        bottom_height = xn[j][0]*alpha
        depth = amb_height - bottom_height
        height = xn[j][1] - amb_height
        k[j] = 2*height*const/strech**2
        if k[j] < kmin : k[j] = kmin
        if k[j] > kmax: k[j] = kmax

        j += 1
    return k

"""
#This section considered varying the mass of the points
#which could be used at points where air begins to mix with water at
#the top of the wave
def get_m(xn,m_min,m_max,min_spacing):
    m = np.zeros(len(xn))
    m.fill(m_max)

    if m_min != m_max:
        spacing = []
        i = 0
        while i < len(xn)-1:
            spacing.append(np.linalg.norm(xn[i]-xn[i+1]))
            i += 1
        all_ctrpts = []
        i = 0
        while i < len(spacing)-1:
            if spacing[i-1] > spacing[i] < spacing[i+1]:
                all_ctrpts.append([i,spacing[i]])
            i += 1
        ctrpts = []
        i = 0
        while i < len(all_ctrpts):
            if all_ctrpts[i][1] < min_spacing:
                ctrpts.append(all_ctrpts[i][0])
            i += 1
        i = 0
        print(ctrpts)
        if len(ctrpts)>1:
            #Quadratic
            while i < len(ctrpts)-1:
                c = -m_min
                min_c = max(ctrpts[i],1)
                max_c = max(ctrpts[i+1],1)
                a = c/(max_c*min_c)
                b = (-c*max_c - c*min_c)/(max_c*max_c)
                j = ctrpts[i]
                while j < ctrpts[i+1]:
                    m[j] = np.abs(a*j**2 - b*j + c)+m_min
                    j += 1
                i += 1
            
            #Gaussian
            while i < len(ctrpts)-1:
                j = ctrpts[i]
                while j < ctrpts[i+1]:
                    m[j] = -(m_max-m_min)*np.exp(1/(ctrpts[i+1]-ctrpts[i])*(j-(ctrpts[i+1]-ctrpts[i]))**2)+m_max
                    j += 1
                i += 1
            
        print(m)
    return m
"""


def verlet(x0,xnm1,v0_horiz,acceleration,a0,m,kmin,kmax,dk,alpha,gamma,LFdt,dt,nsteps,LF):
    """
	This function computes the entire simulation using the Verlet method
    """
    xout = []
    k = np.zeros(len(x0))
    k.fill(kmin)

    #m = np.zeros(len(x0))
    #m.fill(m_max)
    if LF:
        v0_vert = (x0-xnm1)/(LFdt*1)
       
        v0 = v0_vert+v0_horiz
        
    else:
        v0 = v0_horiz
    xout.append(x0)

    vn = v0

    xn = x0
    a,potl_energy = acceleration(xn,vn,a0,m,k,gamma)
    vmag = np.array([sqrt(v[0]**2+v[1]**2) for v in v0])
    kin_energy = 0.5*m*vmag**2
    
    xnp1 = x0 + v0*dt + 0.5*a*dt**2
    xnm1 = xn
    xn = xnp1
    xout.append(xnm1)
    energy = [potl_energy+kin_energy]

    peak_vel = []
    
    j = 1
    while j < nsteps:
        total_energy = 0
        if kmin != kmax:
            if j == 1:
                print("-- Variable K --")
            #k = getk_th(xn,kmin,kmax)
            #k = getk_d(xn,kmin,kmax,alpha,dk)
            const = .1
            k = getk_e(xn,kmin,kmax,a0,alpha,const)
        elif j == 1:
            print("-- K Constant --")

        #For variable mass
        min_spacing = .1
        #m = get_m(xn,m_min,m_max,min_spacing)

        a,potl_energy = acceleration(xn,vn,a0,m,k,gamma)

        xnp1 = 2*xn - xnm1 + dt**2 * a #update the position
        
        vn = (xnp1-xnm1)/(2*dt)

        vmag = np.array([np.linalg.norm(v) for v in vn])

        kin_energy = 0.5*m*vmag**2
        energy.append(kin_energy+potl_energy)

        xnm1 = xn
        xn = xnp1

        xout.append(xn)
        #yout.append(xn[1])
        if (j/nsteps*100) % 10 == 0: print(str(int(j/nsteps*100))+"%") #prints percentage complete
        j+=1
        #out.append([xout,yout])

    #velocity checking
    """
    v_first_index = 100
    delta = 50
    v_last_index = v_first_index + delta

    print("velocity:",(max_h(xout[v_last_index])-max_h(xout[v_first_index]))/(delta*dt))
    """
    print("Calculation Complete.")

    e_dev = (np.std(energy)/np.mean(energy))*100

    if gamma == 0:
        print("Energy Error (stdev/mean):"+str(e_dev)+"%")
    return xout,energy

def max_h(xn):
    data_right = xn[len(xn)//2:]
    temp_max = 0
    i = 0
    while i < len(data_right)-1:
        if (data_right[i][1] >= temp_max) and (data_right[i+1][1] < data_right[i][1]):
            temp_max = data_right[i][0]
        i += 1
    return temp_max

def data_to_pairs(data):
	"""
	Some data is in two lists, [[x1,x2,...],[y1,y2,...]]
	Sometimes it's way easier to work with [[x1,y1],[x2,y2],...]
	"""
    out = []
    i = 0
    while i < len(data[0]):
        x = data[0][i]
        y = data[1][i]
        out.append(np.array([x,y]))
        i+=1
    return np.array(out)

def data_to_xy(data):
	"""
	Some data is in ordered pairs [[x1,y1],[x2,y2],...]
	
	Sometimes it's way easier to work with two lists, [[x1,x2,...],[y1,y2,...]]
	"""
    xout = []
    yout = []
    for pos in data:
        xout.append(pos[0])
        yout.append(pos[1])
    return np.array([xout,yout])