#%%
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# import numpy as np


# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


# x_rand = np.random.randint(1,61,60)
# y_rand = np.random.randint(1,61,60)


# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# app.layout = html.Div(children=[

#     html.H1(children='Hello Dash'),

#     html.Div(children=[

#         dcc.Graph(
#             id='example-graph',
#             figure={
#                 'data': [
#                     {'x': [1, 2, 3], 'y': [4, 1, 2], 'type': 'markers', 'name': 'SF'},
#                     {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'markers', 'name': u'Montréal'},
#                 ],
#                 'layout': {
#                     'title': 'Dash Data Visualization'
#                 }}),

#     ], style={'display': 'inline-block'}),

#     html.Div(children=[

#         dcc.Graph(
#                 id='example-graph2',
#                 figure={
#                     'data': [
#                         {'x': [5, 6, 3], 'y': [14, 21, 21], 'type': 'bar', 'name': 'SF'},
#                         {'x': [3, 5, 7], 'y': [12, 43, 54], 'type': 'bar', 'name': u'Montréal'},
#                     ],
#                     'layout': {
#                         'title': 'Dash Data Visualization222'
#                     }}),

#     ], style={'display': 'inline-block'}),

# ], style={'width': '100%', 'display': 'inline-block'})




# if __name__=="__main__":
#     app.run_server(debug=True)

#%%
# import dash
# from dash.dependencies import Output, Input
# import dash_core_components as dcc
# import dash_html_components as html
# import plotly
# import random
# import plotly.graph_objs as go
# from collections import deque


# X = deque(maxlen = 20)
# X.append(1)


# Y = deque(maxlen = 20)
# Y.append(1)

# app = dash.Dash(__name__)

# app.layout = html.Div(
#     [    
#         dcc.Graph(id = 'live-graph',
#                   animate = True),
#         dcc.Interval(
#             id = 'graph-update',
#             interval = 1000,
#             n_intervals = 0
#         ),
#     ]
# )

# @app.callback(
#     Output('live-graph', 'figure'),
#     [ Input('graph-update', 'n_intervals') ]
# )
# def update_graph_scatter(n):
#     X.append(X[-1]+1)
#     Y.append(Y[-1]+Y[-1] * random.uniform(-0.1,0.1))

#     data = plotly.graph_objs.Scatter(
#             x=list(X),
#             y=list(Y),
#             name='Scatter',
#             mode= 'lines+markers'
#     )

#     return {'data': [data],
#             'layout' : go.Layout(xaxis=dict(
#                     range=[min(X),max(X)]),yaxis = 
#                     dict(range = [min(Y),max(Y)]),
#                     )}


# if __name__ == '__main__':
#     app.run_server()



#%%

# import dash 
# import dash_core_components as dss
# import dash_html_components as html

# app = dash.Dash()

# app.layout = html.Div(children=[html.H1('Dash tutorials')])

# if __name__ == '__main__':
#     app.run_server(debug=True)

#%%
import dash 
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq


app = dash.Dash()

# app.layout = html.Div(children=[html.H1('Dash tutorialsssss'),
# dcc.Graph(id='example',
#         figure={
#             'data':[

#                 {'x':[1,2,3,4,5],'y':[3,5,6,9,2],'type':'point','name':'tp'}
#             ],
#             'layout':{'title':"Basics Dash"}
#                     }

# )

# ])

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




#change these for every experiment 
# folder = base_dir+'exp-1/' 
saveFig = False #Toggle Switch
fileFormat ='png' #file Format
plotN = 0 #this variable picks the last "plotN" history for plotting 
nb = 3 #slider from 1 to 10
nn = 3 #slider from 1 to 10
iterations = 500000 #RL iterations to set numbers only 
GC =0.01 #slider
DC =0.01 #slider
epsilon_mod = 3000
K = 25000

# behavior = Behavior(nNutrients=nn,stateSlicer = 11,actionSlicer = 2)
# brain = Brain(behavior)
# gut = Gut(nBacteria = nb)

# nrows = 2 if nn > 3 else 1 # change this if nutirents are more than 4
# ncols = gut.nBacteria//nrows

# stepSize = 1

# counter = 0
# n = iterations//stepSize
# gut.gc = 0.01
# gut.dc = 0.01
# gut.contribution = np.array([1,0,0])
# epsilon_mod = 3000
# gut.K = 25000

# # below parameters to pick appropriate data points for plotting only.
# choice = 'random'
# plotDataPoints = 500
# start,stop,step = 0,5,1
# sel = generatePlotSel(choice,n = plotDataPoints,iterations = iterations)  #check code definition to send proper variables


app.layout = html.Div([
    dcc.Input(id='input', value='Enter something here!', type='text'),
    html.Div(id='output'),
    dcc.Graph(id='example',
        figure={
            'data':[

                {'x':[1,2,3,4,5],'y':[3,5,6,9,2],'type':'point','name':'tp'}
            ],
            'layout':{'title':"Basics Dash"}
                    }

)
])

@app.callback(
    Output(component_id='output', component_property='children'),
    [Input(component_id='input', component_property='value')]
)
def update_value(input_data):
    return 'Input: "{}"'.format(input_data)

if __name__ == '__main__':
    app.run_server(debug=True)

