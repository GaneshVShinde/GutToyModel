#%%
#@title
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import random
import time
from tqdm import tqdm

from datetime import datetime
import os,errno
# %%
#@title
def createDir(dir):
  try:
      os.makedirs(dir)
      print("created directory",dir)
  except OSError as e:
      print(e)
    #   if e.errno == errno.EEXIST:
    #       print('Directory ',dir,'already exists.')
    #   else:
    #       raise  
# %%
#@title
def writeParameters():
  if saveFig:
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    createDir(folder)
    
    params = """ SIMULATION RAN ON : {dt}
    number of iterations = {0}
    first slice for graph plotting, plotN = {1}
    step size, history recorded at every step size  = {2}
    
    
    plot selection variables : choice = {ps1}, number of data points for representation = {ps2}
    in case of a random slice of history:start = {ps3}, stop = {ps4}, step = {ps5}
    
    
    BEHAVIOUR PARAMETERS 
    =======================
    # nutrients = {b1}
    state slicer = {b2}
    action slicer = {b3}

    shapes of stm, state space, action space = {b4},{b5},{b6}


    GUT PARAMETERS
    ===============
    Growth constant = {g1}
    Decay constant = {g2}
    K = {g3}
    per bacteria contribution = {g4}

    BRAIN PARAMETERS
    ==================
    alpha = {bb1}
    gamma = {bb2}
    epsilon = {bb3}

    """.format(iterations,plotN,stepSize
    ,b1 = behavior.nNutrients,b2 = behavior.stateSlicer,b3 = behavior.actionSlicer
    ,b4 = behavior.stm.shape,b5 = len(behavior.stateSet),b6 = len(behavior.actionSet)
    ,b7 = behavior.stateSet,b8 = behavior.actionSet
    ,g1 = gut.gc,g2 = gut.dc,g3 = gut.K,g4 = gut.contribution
    ,bb1 = brain.alpha,bb2 = brain.gamma, bb3 = brain.epsilon
    ,ps1 = choice, ps2 = plotDataPoints, ps3 = start, ps4 = stop, ps5 = step
    ,dt = dt_string
    )

    with open(folder+"parameters.txt", "a") as file:
        file.write(params)

# %%
#@title
def generatePlotSel(choice = 'random',**kwargs):
    sel = np.arange(5)
    if choice == 'random':
        n = kwargs.get('n',5)
        randSel = np.random.randint(0,iterations,n)
        randSel.sort()
        sel =  randSel
    elif choice == 'all':
        sel = np.arange(iterations)
    elif choice == 'custom':
        start = kwargs.get('start',0)
        stop = kwargs.get('stop',5)
        step = kwargs.get('step',1)
        sel = np.arange(start,stop,step)
    
    else:
        print("first five selection only.")
    return sel

#%%
#@title
   
class Brain:
    def __init__(self, behavior, **kwargs):

        # Params
        self.alpha = kwargs.get('alpha', 0.1)
        self.gamma = kwargs.get('gamma', 0.05)
        self.epsilon = kwargs.get('epsilon', 0.1)
        
        self.time = kwargs.get('time', 1)  #for debugging

        # for convenience
        self.nActions = len(behavior.actionSet)
        self.nStates = len(behavior.stateSet)

        # To be computed
        self.qTable = np.zeros((self.nStates,self.nActions))
        self.oStateInd = behavior.stateInd
        self.oActionInd = None


    def do(self,stateInd):
        self.oStateInd= stateInd
        if(np.random.random() > self.epsilon):
            actionInd = self.exploit(stateInd)
        else:
            actionInd = self.explore()

        self.oActionInd = actionInd
        return self.oActionInd

        

    def learn(self,newstate, reward):
        
        old_q = self.qTable[self.oStateInd,self.oActionInd]
        maxFutureQ = max(self.qTable[newstate, :])

        self.qTable[self.oStateInd, self.oActionInd] = old_q + self.alpha*(reward +                 self.gamma*maxFutureQ - old_q)

        
           
    def exploit(self,stateInd):
        possibleActionValues = self.qTable[stateInd, :]
        maxActionInd = np.where(possibleActionValues == max(possibleActionValues))[0]
        return np.random.choice(maxActionInd)
        
        
    def explore(self):
        return np.random.choice(range(self.nActions))

#%%
#@title
class Behavior:
    def __init__(self,initState=None,nNutrients = 3, **kwargs):


        self.nNutrients = nNutrients

    # Create state and action sets
        self.stateSlicer =  kwargs.get('stateSlicer',5)
        self.actionSlicer =  kwargs.get('actionSlicer', 2)

        tempState = np.linspace(0, 1, self.stateSlicer).round(2)
        #to include space for each nutrient, dynamically add state space for each nutrient
        tempList = np.tile(tempState,(nNutrients,1)) 
        self.stateSet = list(product(*tempList)) 
        
        temp_t = tempState[1]-tempState[0]
        tempAction = np.linspace(-temp_t,temp_t,self.actionSlicer)
        tempList = np.tile(tempAction,(nNutrients,1)) 
        self.actionSet = list(product(*tempList))


        if not initState:
            self.stateInd = np.random.choice(range(len(self.stateSet)))
            print("""randomly initialised to {0} state""".format(self.stateSet[self.stateInd]))
        else:
            state =  initState   #Initializing
            self.stateInd = self.findStateIndex(state)
            print("""initialised to  {0}""".format(state))


        self.stm = np.zeros((len(self.stateSet), len(self.actionSet)))
        self.computeStateTransitions()


        self.output = np.zeros((self.nNutrients))


    def computeStateTransitions(self):
        tic = time.perf_counter()

        for (si,_),(ai,_) in product(enumerate(self.stateSet),enumerate(self.actionSet)):
            nsi = self.updateState(si,ai) 
            self.stm[si,ai] = nsi
        toc = time.perf_counter()

        print("""total time to compute state transition matrix = {0:.2f} seconds""".format(toc-tic))


    def findStateIndex(self,s):
        return np.where((self.stateSet == s).all(axis=1))[0][0]           


    def updateState(self,si,ai):
        s = self.stateSet[si]
        a = self.actionSet[ai]
        #new state defined by behavior based upon action chosen by brain
        ns = np.clip(np.array(s)+np.array(a),0,1).round(2) 
        ind = self.findStateIndex(ns)
        return ind

    def ingestNutrients(self):
        # logic 1: nutrients are operationalised to a binary variable stating the existence of the nutrients -i.e., whether the specified nutrient was ingested or not?
        # self.output = np.random.binomial(1,self.stateSet[self.stateInd])
        
        # logic 2: nutrients are real numbers from (0,100) from a normal distribution with mean = current state of the brain [AND a constant variance for all nutrients]
        var = 0.1
        self.output = np.random.normal(self.stateSet[self.stateInd],var)*100
        

# %%
#@title
class Gut:
    def __init__(self,initPop = None,nBacteria = 3,**kwargs):
        

        self.nBacteria = nBacteria
        self.pop = np.random.choice(np.arange(100,200),self.nBacteria)
        self.init_pop = self.pop
        self.gc = kwargs.get('gc',0.1)
        self.dc = kwargs.get('dc',0.1)
        self.K = kwargs.get('K',25000)
        self.contribution = kwargs.get('contribution',np.ones((nBacteria)))
            
    ## Population-based implementation
    def updatePopulation(self, nutrients):
        self.pop = self.pop*(1 + self.gc*nutrients*(1- np.sum(self.pop)/self.K) - self.dc*(self.nBacteria*self.pop/self.K))


    def generateReward(self,nutrients):

        self.reward = np.sum(self.pop*self.contribution*nutrients)
        self.reward = self.reward/(self.gc*self.K/self.dc)
    
# %%

destination = ''

if destination == 'drive':
    # run the below code if running in Colab and you wish to save file in the drive.
  from google.colab import drive
  drive.mount('/content/gdrive')
  base_dir = '/content/gdrive/MyDrive/Colab Notebooks/'
else:
  base_dir = ''
base_dir = base_dir + '../simulations/test-code'
createDir(base_dir)

# %%

#change these for every experiment 
folder = base_dir+'exp1/' 
saveFig = True
fileFormat ='png'
plotN = 0 #this variable picks the last "plotN" history for plotting 
nb = 3
nn = 3

behavior = Behavior(nNutrients=nn,stateSlicer = 6,actionSlicer = 2)
brain = Brain(behavior)
gut = Gut(nBacteria = 3)

nrows = 2 if gut.nBacteria > 3 else 1 # change this if nutirents are more than 4
ncols = gut.nBacteria//nrows

stepSize = 1
iterations =  500000
counter = 0
n = iterations//stepSize
gut.gc = 0.01
gut.dc = 0.01
gut.contribution = np.array([1,0,1])
epsilon_mod = 3000
gut.K = 25000

# below parameters to pick appropriate data points for plotting only.
choice = 'random'
plotDataPoints = 500
start,stop,step = 0,5,1
sel = generatePlotSel(choice,n = plotDataPoints,iterations = iterations)  #check code definition to send proper variables


# %%
%%time

# instead of recording system state at fixed intervals, we can also record history at random timesteps
# randSel = np.random.randint(0,iterations,n)
# randSel.sort()
np.set_printoptions(suppress = True)
behaviorHistory = np.zeros((iterations//stepSize,behavior.nNutrients))
gutPopHistory = np.zeros((iterations//stepSize,gut.nBacteria))
rewardHistory = np.zeros((iterations//stepSize))
stateHistory = np.zeros((n,behavior.nNutrients))
actionHistory = np.zeros((n,behavior.nNutrients))
idx=0
for t in tqdm(range(iterations)):
    

    # find best possible action in brain using RL
    actionInd = brain.do(behavior.stateInd)
    
    # update behavior state as a result of action chosen by brain
    oldInd = behavior.stateInd
    newStateInd = behavior.updateState(behavior.stateInd,actionInd)
    behavior.stateInd = newStateInd

    #find the reward elicited by gut
    behavior.ingestNutrients()
    gut.generateReward(behavior.output)    
    gut.updatePopulation(behavior.output)
    
    brain.learn(newStateInd,gut.reward)
    brain.time = t

    if t%epsilon_mod ==0:
        brain.epsilon*=0.99

    if t%stepSize==0:
        behaviorHistory[idx] = behavior.output
        gutPopHistory[idx] = gut.pop
        rewardHistory[idx] = gut.reward
        stateHistory[idx] = behavior.stateSet[oldInd]
        actionHistory[idx] = behavior.actionSet[actionInd]

        idx+=1

    # if idx<n and t==randSel[idx] :
    #     behaviorHistory[idx] = behavior.output
    #     gutPopHistory[idx] = gut.pop
    #     rewardHistory[idx] = gut.reward
    #     stateHistory[idx] = behavior.stateSet[behavior.stateInd]
    #     idx+=1

writeParameters()

#%%

def plotState():
  #@title
  # one scatter plot for state history. 
  # relevant to trace out the population evolution across timesteps.


  X = np.arange(sel.size)
  fig,ax = plt.subplots(nrows,ncols,sharex = True, sharey = True,figsize = (15,6))
  cm = plt.get_cmap("tab10")
  ax = ax.ravel()

  for i in range(nb):
    ax[i].scatter(X,stateHistory[sel,i],color = cm(i))
    ax[i].set_ylabel("cravings")
    ax[i].set_xlabel("time")
    ax[i].text(0.5, 0.5, gut.contribution[i], horizontalalignment='center', verticalalignment='center', transform=ax[i].transAxes)
    ax[i].set_title("""contribution = {0}, mean state = {1:.2f}""".format(gut.contribution[i], stateHistory[:,i].mean()))

  fig.text(0.3,0.9,"""Individual state history
  gc = {0:.2f},dc = {1:.2f}; exploration_decay = {2}
  """.format(gut.gc,gut.dc,epsilon_mod))
  if saveFig:
    plt.savefig(folder+'scatter-state'+str(counter+1)+'.'+fileFormat,bbox_inches = 'tight',format= fileFormat)    
  # plt.close()
plotState()
# %%
import dash 
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash()

app.layout = html.Div(children=[html.H1('Dash tutorialsssss'),
dcc.Graph(id='example',
        figure={
            'data':[

                {'x':[1,2,3,4,5],'y':[3,5,6,9,2],'type':'line','name':'tp'}
            ],
            'layout':{'title':"Basics Dash"}
                    }

)

])

if __name__ == '__main__':
    app.run_server(debug=True)


# %%
