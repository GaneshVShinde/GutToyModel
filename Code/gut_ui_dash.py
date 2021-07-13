import dash
import dash_daq as daq
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_core_components as dcc
import plotly.graph_objs as go
import plotly
import numpy as np
import pymongo
import subprocess
import psutil
import signal
import os
import traceback




mongo_url = "mongodb://localhost:27017/"

mongo_client = pymongo.MongoClient(mongo_url)
db_gut=mongo_client["gut"]

controls_db= db_gut["controls"]

gutHistory_db=db_gut["gutHistory"]
stateHistory_db=db_gut["stateHistory"]
rewards_db=db_gut["rewardHistory"]
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


saveFig = False #Toggle Switch
fileFormat ='png' #file Format
figure_location=''

plotN = 0 #this variable picks the last "plotN" history for plotting 
nb = 2 #slider from 1 to 10
nn = 2 #slider from 1 to 10
iterations = 500000 #RL iterations to set numbers only 
GC =0.01 #slider
DC =0.01 #slider
epsilon_mod = 3000
K = 25000

app = dash.Dash("my_test")

controls_dict=dict()
controls_dict['id']=1
controls_dict["saveFig"]=False
controls_dict["nb"]=2
controls_dict["nn"]=2
controls_dict["iterations"]=50000
controls_dict["GC"]=0.01
controls_dict["DC"]=0.01
controls_dict["epsilon_mod"]=3000
controls_dict["K"]=25000


# app.layout = html.Div([
#     daq.BooleanSwitch(id='my-boolean-switch',on=False,label="Save Figure" , style={'display': 'inline-block', 'height': '50px'}),
#     html.Div(id='boolean-switch-output',style={'margin-top': 20}),
#     # dcc.Input(id='folderInput', value='paste figure location', type='text'),
#     # html.Div(id='output'),
    
#     daq.Slider(id='nb',handleLabel={"label": "nb"},    min=1,max=10,step=1,value=3,),
#     html.Div(id='nbtext',style={'margin-top': 5}),

#     daq.Slider(id='nn',handleLabel={"label": "nn"},min=1,max=10,step=1,value=3),
#     html.Div(id='nntext',style={'margin-top': 5,'display': 'inline-block'}),
    
#     daq.Slider(id='gc',handleLabel={"label": "GC"},min=1,max=10,step=1,value=3),
#     html.Div(id='gctext',style={'margin-top': 5,'display': 'inline-block'}),

#     daq.Slider(id='dc',handleLabel={"label": "DC"},min=1,max=10,step=1,value=3),
#     html.Div(id='dctext',style={'margin-top': 5,'display': 'inline-block'}),

#     daq.NumericInput(id='iterations',value=50000, label='Iterations',size=120,style={ 'height': '50px'}),
#     html.Div(id='itertext',style={'margin-top': 5}),

#     daq.BooleanSwitch(id='start-stop-switch',on=False , style={'display': 'inline-block', 'height': '50px'}),
#     html.Div(id='start-stop-switch-output',style={'margin-top': 1,'display': 'inline-block'}),

#     dcc.Interval(id='population-interval-component',interval=5*1000, n_intervals=0),
#     dcc.Graph(id='population-graph'),
    
#     dcc.Interval(id='cravings-interval-component',interval=5*1000, n_intervals=0),
#     dcc.Graph(id='cravings-graph'),

#     dcc.Interval(id='rewards-interval-component',interval=5*1000, n_intervals=0),
#     dcc.Graph(id='rewards-graph'),

#     #html.Div(dcc.Graph(id='empty', figure={'data': []}), style={'display': 'none'})
# ],className="container",style={'width':'98%','margin-left':10,'margin-right':10,'max-width':50000,'columnCount': 2})



app.layout = html.Div([
    html.Div([daq.BooleanSwitch(id='my-boolean-switch',on=False,label="Save Figure" , style={'display': 'inline-block', 'height': '50px'}),
    html.Div(id='boolean-switch-output',style={'margin-top': 20}),
    # dcc.Input(id='folderInput', value='paste figure location', type='text'),
    # html.Div(id='output'),
        # dcc.Slider(id='nb',min=1,max=20,step=1,value=3),
    
        daq.Slider(id='nb',handleLabel={"label": "nb"},    min=1,max=10,step=1,value=3,),
        html.Div(id='nbtext',style={'margin-top': 5,'margin-bottom':-5}),
        html.Div(id='nbtext2',style={'margin-top': 25,'margin-bottom':-5}),

        # daq.Slider(id='nn',handleLabel={"label": "nn"},min=1,max=10,step=1,value=3),
        # html.Div(id='nntext',style={'margin-top': 5,}),
        # html.Div(id='nntext2',style={'margin-top': 250,}),


        daq.Slider(id='gc',handleLabel={"label": "GC"},min=0.01,max=1,step=0.01,value=0.01),
        html.Div(id='gctext',style={'margin-top': 5,'display': 'inline-block'}),
        html.Div(id='gctext2',style={'margin-top': 250,}),

        daq.Slider(id='dc',handleLabel={"label": "DC"},min=0.01,max=1,step=0.01,value=0.01),
        html.Div(id='dctext',style={'margin-top': 10,'display': 'inline-block'}),
        html.Div(id='dctext2',style={'margin-top': 20,}),


        


        daq.NumericInput(id='iterations',min=1000,max=1000000,value=50000, label='Iterations',size=120,style={ 'height': '50px','display': 'inline-block'}),
        html.Div(id='itertext',style={'margin-top': 5}),

        daq.BooleanSwitch(id='start-stop-switch',on=False , style={'margin-top': 10,'display': 'inline-block', 'height': '50px'}),
        html.Div(id='start-stop-switch-output',style={'margin-top': 1,'display': 'inline-block'})
    ],style={'columnCount': 2}),

    html.Div([
        dcc.Interval(id='population-interval-component',interval=5*1000, n_intervals=0),
        dcc.Graph(id='population-graph'),
        
        dcc.Interval(id='cravings-interval-component',interval=5*1000, n_intervals=0),
        dcc.Graph(id='cravings-graph'),

        dcc.Interval(id='rewards-interval-component',interval=5*1000, n_intervals=0),
        dcc.Graph(id='rewards-graph'),

    #html.Div(dcc.Graph(id='empty', figure={'data': []}), style={'display': 'none'})
],className="container",style={'width':'98%','margin-top': 100,'margin-left':10,'margin-right':10,'max-width':50000})])




@app.callback(
    dash.dependencies.Output('boolean-switch-output', 'children'),
     [dash.dependencies.Input('my-boolean-switch', 'on')])
def update_output(on):
    saveFig= on
    controls_db.update_one({'id':1},{'$set':{'saveFig':on}})
    #print('The switch is {}.'.format(on))
    return ""

pro=0

@app.callback(
    dash.dependencies.Output('start-stop-switch-output', 'children'),
     [dash.dependencies.Input('start-stop-switch', 'on')])
def update_Start_Stop_Button(on):
    global pro
    global prev_len
    global current_len
    
    start_code= on
    if on:
        try:
            prev_len=0

            pro = subprocess.Popen("python ../Code/sim-pipeline.py",stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid) 
            if psutil.pid_exists(pro.pid):
                print("Yepee  added")
        except:
            traceback.print_exc()
            pass


        return "Stop"
    else:
        if pro!=0:
            try:
                os.killpg(os.getpgid(int(pro.pid)), signal.SIGTERM) 
            except:
                traceback.print_exc()
                pass
        return "Start"


@app.callback(
    dash.dependencies.Output('nbtext', 'children'),
    [dash.dependencies.Input('nb', 'value')])
def update_nb_slider(value):
    global nb
    nb=value
    controls_db.update_one({'id':1},{'$set':{'nb':nb}})
    print("nb-updated",nb)
    return 'Number of microbiome: {}'.format(value)


# @app.callback(
#     dash.dependencies.Output('nntext', 'children'),
#     [dash.dependencies.Input('nn', 'value')])
# def update_nn_slider(value):
#     global nn
#     nn=value
#     controls_db.update_one({'id':1},{'$set':{'nn':nn}})
#     return ' {}'.format(value)

@app.callback(
    dash.dependencies.Output('gctext', 'children'),
    [dash.dependencies.Input('gc', 'value')])
def update_nn_slider(value):
    global gc
    gc=value
    controls_db.update_one({'id':1},{'$set':{'gc':nn}})
    return 'Groth Factor:  {}'.format(value)

@app.callback(
    dash.dependencies.Output('dctext', 'children'),
    [dash.dependencies.Input('dc', 'value')])
def update_nn_slider(value):
    global gc
    gc=value
    controls_db.update_one({'id':1},{'$set':{'dc':nn}})
    return 'Decay Factor:  {}'.format(value)


@app.callback(
    dash.dependencies.Output('itertext', 'children'),
    [dash.dependencies.Input('iterations', 'value')])
def update_iterations(value):
    iterations=value
    controls_db.update_one({'id':1},{'$set':{'iterations':iterations}})
    print(value)
    return ''

lst=[]
y1=[]
y2=[]
y3=[]
x=[]
prev_len=0
current_len=0

# Multiple components can update everytime interval gets fired.
@app.callback(Output('population-graph', 'figure'),
              Input('population-interval-component', 'n_intervals'))
def update_population_graph_live(n):

    lst=list(gutHistory_db.find({}))
    pop_mat=np.array([np.array(list(l.items())[1][1])for l in lst])



    # Create the graph with subplots
    col =3
    row=int(pop_mat.shape[1]/col)
    fig = plotly.tools.make_subplots(rows=int(pop_mat.shape[1]/col), cols=col, vertical_spacing=0.2)
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 30, 't': 10
    }
    fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}

    col_n=0
    for r in range(1,row+1):
        for c in range(1,col+1):
            fig.append_trace(go.Scattergl(x=list(range(pop_mat.shape[0])),y=pop_mat[:,col_n],
                                name="mb:"+str(col_n+1),mode='lines'),r,c)
            col_n+=1

    return fig


#cravings graph
@app.callback(Output('cravings-graph', 'figure'),
              Input('cravings-interval-component', 'n_intervals'))
def update_cravings_graph_live(n):

    lst=list(stateHistory_db.find({}))
    state_mat=np.array([np.array(list(l.items())[1][1])for l in lst])



    # Create the graph with subplots
    col =3
    row=int(state_mat.shape[1]/col)
    print(row)
    fig = plotly.tools.make_subplots(rows=int(state_mat.shape[1]/col), cols=col, vertical_spacing=0.2)
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 30, 't': 10
    }
    fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}

    col_n=0
    for r in range(1,row+1):
        for c in range(1,col+1):
            fig.append_trace(go.Scattergl(x=list(range(state_mat.shape[0])),y=state_mat[:,col_n],
                                name="cravings :"+str(col_n+1),mode='markers'),r,c)
            col_n+=1

    return fig


@app.callback(Output('rewards-graph', 'figure'),
              Input('rewards-interval-component', 'n_intervals'))
def update_cravings_graph_live(n):

    rewards_l=list(rewards_db.find({}))
    rewards_mat=np.array([list(re.items())[1][1] for re in rewards_l])  

    fig = plotly.tools.make_subplots(rows=1, cols=1, vertical_spacing=0.2)
    fig['layout']['margin'] = {
        'l': 30, 'r': 10, 'b': 30, 't': 10
    }
    fig['layout']['legend'] = {'x': 0, 'y': 1, 'xanchor': 'left'}
    fig.append_trace(go.Scattergl(x=list(range(rewards_mat.shape[0])),y=rewards_mat,
                                name="reward",mode='markers'),1,1)


    return fig


# @app.callback(Output('container', 'children'), [Input('interval-component', 'n_intervals')])
# def display_graphs(n_intervals):
#     global nb
#     global lst
#     global y1
#     global y2
#     global y3
#     global x
#     global prev_len
#     global current_len
#     graphs = []
#     print(nb)
#     #prev_len=len(lst)
#     lst=list(gutHistory_db.find({}))
#     print(len(lst),prev_len)
#     if prev_len!=len(lst):
#         for l in lst[prev_len:len(lst)]:
#             k=list(l.keys())[1]
#             x.append(int(k))
#             y1.append(l[k][0])
#             y2.append(l[k][1])
#             y3.append([l[k][2]])
#         prev_len=len(lst)

#     print(len(y3))

#     if nb>2:
#         class_choice = 'col s12 m6 l4'
#     elif nb == 2:
#         class_choice = 'col s12 m6 l6'
#     else:
#         class_choice = 'col s12'
#     for i,y in enumerate([y1,y2,y3]):
#         graphs.append(dcc.Graph(
#         id='life-exp-vs-gdp',
#         figure={
#             'data': [
#                 go.Scatter(
#                     x=x,
#                     y=y,
#                     text="Graph:"+str(i),
#                     mode='markers',
#                     opacity=0.8,
#                     marker={
#                         'size': 15,
#                         'line': {'width': 0.5, 'color': 'white'}
#                     },
#                     name=i
#                 )
#             ],
#             'layout': go.Layout(
#                 xaxis={ 'title': 'X-random'},
#                 yaxis={'title': 'Y-random'},
#                 margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
#                 legend={'x': 0, 'y': 1},
#                 hovermode='closest'
#             )
#         }
#     ))
#     return html.Div(graphs,className=class_choice)




# @app.callback(
#     Output(component_id='output', component_property='children'),
#     [Input(component_id='folderInput', component_property='value')]
# )
# def update_value(input_data):
#     figure_location=input_data
#     return ""



def save_figure_toggle():
    pass

def update_population():
    pass

def update_states():
    pass

def update_behaviour():
    pass

def plotQtable():
    pass


external_css = ["https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/css/materialize.min.css"]
for css in external_css:
    app.css.append_css({"external_url": css})

external_js = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/js/materialize.min.js']
for js in external_css:
    app.scripts.append_script({'external_url': js})

if __name__ == '__main__':
    controls_db.remove()
    controls_db.insert_one(controls_dict)
    app.run_server(debug=True)