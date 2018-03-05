import numpy as np

#####
##
## Lax Friedrichs Method
##
#####
def updateLF(ineta,ink,ds,dq,alpha,stable):
    """
    Updates the eta height map to the next time step
    used in runLF
    """
    Nlen = len(ineta)
    #Start with zeros to be later filled with data (instead of append since j+1 is required element)
    outeta = [0 for i in range(0,Nlen)]
    outk = [0 for i in range(0,Nlen)]
    outv = [0 for i in range(0,Nlen)]
    
    # The Boundary Conditions
    outk[0] = ink[1]
    outk[Nlen-1] = ink[Nlen-2] # Derivative condition (height can vary)

    outeta[0] = ineta[1]
    outeta[Nlen-1] = ineta[Nlen-2] # Derivative condition (height can vary)

    j = 1
    while j <= Nlen-2:
        #Updates
        outeta[j] = .5*(ineta[j+1]+ineta[j-1])-ds/(2*dq)*(ink[j+1]*(1-alpha*(j+1)*dq/(ineta[j+1]))-ink[j-1]*(1-alpha*(j-1)*dq/(ineta[j - 1])))
        outk[j] = 1/(1+alpha*j*dq/(outeta[j]))*((.5*(ink[j+1]*(1+alpha*(j+1)*dq/(ineta[j+1]))+ink[j-1]*(1+alpha*(j-1)*dq/(ineta[j - 1]) ))-(ds/(2*dq))*(ink[j+1]**2/(ineta[j+1])-ink[j-1]**2/(ineta[j-1])+.5 *(ineta[j+1])**2-.5*(ineta[j-1])**2) )+ds/(2*dq)*(ink[j+1]/(ineta[j+1])-ink[j-1]/(ineta[j-1]))*2.*alpha*j*dq*ink[j]/(ineta[j]) +(ink[j]*alpha/ineta[j])**2)
        
        #Update of k for loosened restriction of p
        #p = 0.002
        #outk[j] = (1/(1-(alpha*j*dq/outeta[j])))*(ink[j]*(1-alpha*j*dq/ineta[j])- (ds/(2*dq))*(((ink[j+1]**2/ineta[j+1])+0.5*ineta[j+1]**2-(ink[j+1]**2*alpha*j*dq/ineta[j+1]**2))-((ink[j-1]**2/ineta[j-1])+0.5*ineta[j-1]**2-(ink[j-1]**2*alpha*j*dq/ineta[j-1]**2)))- (2*ink[j]*alpha**2*ds/(ineta[j]*(2*p+1)))*((1/dq)*(ink[j+1]/ineta[j+1]-ink[j-1]/ineta[j-1]))*(ineta[j]-alpha*j*dq)+ink[j]*p/ineta[j])
        


        outv[j] = outk[j]/outeta[j]

        #Check for Stability
        if (outv[j]/outeta[j])* ds > dq:
            stable = False
        j += 1
    return outeta,outk,outv,stable

def runLF(eta0,k0,ds,dq,Nsteps,alpha,loud=True):
    """
    Runs the LF simulation, using updateLF for each time step
    """
    stable = True
    # Builds the initial lists of zeros
    row = [0 for i in range(0,Nsteps)]
    etavals = [row for i in range(0,Nsteps)]
    kvals = [row for i in range(0,Nsteps)]
    vvals = [row for i in range(0,Nsteps)]

    # apply the input initial conditions
    etavals[0] = eta0
    kvals[0] = k0
    
    # run the model
    n = 1
    while n <= Nsteps-1:
        if loud:
            if (n/Nsteps*100) % 10 == 0: print(str(int(n/Nsteps*100))+"%")
        etavals[n],kvals[n],vvals[n],stable = updateLF(etavals[n-1],kvals[n-1],ds,dq,alpha,stable)
        n += 1
    return etavals,kvals,vvals,stable

#####
##
## Convert h list to (x,y) list 
##
#####

def data_to_xy(etadata,dq):
    """
    Given some data of heights, where the list index 
    determines the position on the x axis, returns a 
    list of x values and a list of y values for each time step

    """
    out = []
    for eta in etadata:
        size = np.size(eta)
        outx = [i*dq for i in np.arange(0,size)]
        outy = [eta[i] for i in np.arange(0,size)]
        out.append([outx,outy])
    return np.array(out)

#####
##
## Run the LF, turn to x,y
##
#####



def doLF(alpha,L,dq,ds,Nsteps,eta0,k0,loud=True):
    """
    This function sets up and runs the whole thing to be used in another file
    as a single function call
    """
    if loud:
        print("Calculating Solution...")

    outeta,outk,outv,stable = runLF(eta0,k0,ds,dq,Nsteps,alpha,loud)
    if loud:
        print("100% -- Calculation Complete.")

    if stable == False:
        print("\n\n\nWARNING \nThis solution is not stable per Lax-Friedrichs Reqs.\n(v/h)*ds<dq\n\n")
    
    xyetadata = data_to_xy(outeta,dq)
    xykdata = data_to_xy(outk,dq)
    xyvdata = data_to_xy(outv,dq)
    return outeta,xyetadata,outv,xyvdata
