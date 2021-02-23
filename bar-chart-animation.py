
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

fig = plt.figure()
global barcollection


def initialise():
    global barcollection
    
    n = 5
    x = np.arange(n)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    craving_prob = ['nutrient_'+str(i) for i in range(n)]

    ax = plt.axes(ylim=(0,1))


    barcollection = ax.bar(x,np.zeros(n),color = colors, tick_label = craving_prob)
    return barcollection


def update_bars(i):
    global barcollection
    for j,b in enumerate(barcollection):
        b.set_height(np.random.random())
    return barcollection

anim = animation.FuncAnimation(fig, update_bars,init_func=initialise,
                               frames=100, interval=100, blit=False)


# anim.save('maah'+str(n)+'.mp4')


plt.show()