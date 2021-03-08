
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

# fig = plt.figure()
global barcollection,c1

fig,(ax1,ax2) = plt.subplots(2,1,constrained_layout = True)

def initialise():
    global barcollection,c1
    
    n = 5
    x = np.arange(n)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    craving_prob = ['nutrient_'+str(i) for i in range(n)]

    # ax = plt.axes(ylim=(0,1))
    ax1.set_ylim((0,1))
    ax2.set_xlim((-10,10))
    ax2.set_ylim((-10,10))
    barcollection = ax1.bar(x,np.zeros(n),color = colors, tick_label = craving_prob)
    c1 = plt.Circle(( 0.5 , 0.5 ), 0) 
    return barcollection


def animate_stuff(i):
    add_circles(i)
    update_bars(i)


def add_circles(i):
    global c1
    r1 = 1.1**i%10 if i%10<4 else 1.01**-i%10

    # c2 = plt.Circle((0.3,0.4),r2)
    ax2.set_aspect( 1 ) 
    ax2.add_artist( c1 )
    c1.set_color("red")
    c1.set_radius(r1)

    ax2.set_title("""radius = {0:.2f}""".format(r1),y =0, pad = -10)
    print(r1)
    # plt.title( 'Colored Circle' ) 


def update_bars(i):
    global barcollection
    if i%10 == 0:
        for j,b in enumerate(barcollection):
            b.set_height(np.random.random())
    return barcollection

anim = animation.FuncAnimation(fig, animate_stuff,init_func=initialise,
                               frames=100, interval=100, blit=False)


# anim.save('maah'+str(n)+'.mp4')


plt.show()