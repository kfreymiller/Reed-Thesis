import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import argrelextrema

#####
##
## Animation (Adapted from http://matplotlib.org/examples/animation/basic_example.html)
##
#####
def animate(alpha,L,dq,ds,eta_xy,video_length,save_animation = False):

    print("Loading Animation...")
    Nsteps = len(eta_xy)
    frame_length = (video_length*1000)//Nsteps

    def update_line(num,data,line):
        """
        Updates line to the current frame n, the nth element of the data input
        """

        #Vector Data Update
        """
        vector_data = np.array([v_list[num][int(i/dq)] for i in X]) # Pulls v at each vector q-location
        max_v = np.max(np.absolute(vector_data)) # 1/max_v normalizes the velocity so largest v is always length 1
        if max_v == 0: max_v = 1e-15 # Helps with 1/0 issue
        U = vector_data*(1/max_v) # X coord
        V = 0 # Y coord (effects direction)
        Q.set_UVC(U,V)
        """

        # Wave Update
        line.set_data(data[num]) #Updates the wave-function line

        return line, #Q, 


    fig1,ax=plt.subplots(1,1)
    l, = plt.plot([], [], 'b-')
    """
    #Vector Stuff
    n_vec = 10
    X = np.mgrid[:L:L//n_vec] # Array of evenly spaced points along the graph
    Y = np.zeros(np.size(X))
    U = np.array(np.sin(X))
    V = 0
    Q = ax.quiver(X,Y,U,V,pivot='mid',color='grey',units='inches') # Make the vectors from coords X,Y -> U,V
    max_v = np.max(v_list) # Max velocity to calibrate vector length
    """
    plt.xlim(0, L)
    plt.ylim(-.1, 3)
    plt.xlabel('$x$')
    plt.ylabel('$h$')
    plt.title('')

    plt.plot([0,L],[0,alpha*L],color='black')
    line_ani = animation.FuncAnimation(fig1, update_line, Nsteps, fargs=(eta_xy, l),interval=frame_length, blit=False)

    plt.show()

    if save_animation:
        do_save = input("Save this? [y] or [n] ")
        if do_save == ("y" or "Y"):
            file_name = input("Enter filename without extension: ")
            file_name += ".mp4"
            print('Saving Animation...')
            line_ani.save(file_name)
            print('Animation Saved -- '+ file_name)
