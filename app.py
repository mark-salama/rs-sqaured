import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd


movie_factors = np.loadtxt("movie_factors.csv", delimiter=",")

df = pd.read_csv('movie_dict.csv')

def make_dash_table(df):
    ''' Return a dash definition of an HTML table for a Pandas dataframe '''
    html_row = [html.Td('Rank'),html.Td('Movie'),html.Td('Rating')]
    table = []
    table.append(html.Tr(html_row))

    for index,row in df.iterrows():
        html_row = [html.Td(index+1),html.Td(row.movie_name),html.Td(round(row.avg_rating,2))]
        table.append(html.Tr(html_row))
    return table

app = dash.Dash(__name__)

server = app.server

app.css.append_css({
    'external_url': (
        'https://cdn.rawgit.com/chriddyp/0247653a7c52feb4c48437e1c1837f75'
        '/raw/a68333b876edaf62df2efa7bac0e9b3613258851/dash.css'
    )
})

app.title = 'Movie Recommender'

app.layout = html.Div(children=[

    html.Div(
        [

            html.Img(
                src='https://raw.githubusercontent.com/mark-salama/rs-sqaured/master/logo.png?raw=true',
                # className='one columns',
                style={
                    'height': '125',
                    # 'width': '225',
                    'float': 'left',
                    'position': 'relative',
                    'margin': 25
                },
            ),


        ],
        className='row',
        style={'text-align': 'left', 'margin':25, 'margin-bottom': '15px'}
    ),



    html.Div(
        [   
            html.Div([
                dcc.Dropdown(
                id='movie-selector',

                options=[{'label': rows['movie_name'], 'value': rows['matrix_row']} for index,rows in df.iterrows()],

                multi=True)
            ],
            className='row',
            style={'align': 'left', 'width':600,'margin':25,'margin-bottom': '10px'}

            ),

            html.Div([

                html.P('Year:',
                    style={
                        'display':'inline-block',
                        'verticalAlign': 'top',
                        'marginRight': '10px',
                        'padding': 7
                    }
                ),

                html.Div([
                    dcc.RangeSlider(
                        id='year-slider',
                        min=1970,
                        max=2020,
                        step=5,
                        marks={
                        1970: '1970',
                        1980: '1980',
                        1990: '1990',
                        2000: '2000',
                        2010: '2010',
                        2020: '2020',
                        },
                        value=[1970, 2020]),
                ], style={'width':475, 'display':'inline-block', 'margin': 5, 'marginBottom':30, }),
                
                html.Br([]),
                
                html.P('Rating:',
                    style={
                        'display':'inline-block',
                        'verticalAlign': 'top',
                        'marginRight': '10px',
                        # 'padding':5
                    }
                ),

                html.Div([
                    dcc.RangeSlider(
                        id='rating-slider',
                        min=1,
                        max=5,
                        step=.25,
                        marks={
                        1: '1',
                        2: '2',
                        3: '3',
                        4: '4',
                        5: '5'
                        },
                        value=[2, 5])
                ], 

                style={'width':475, 'display':'inline-block', 'margin': 5,}
                ),],
            
            className='row',
            style={'align': 'right', 'width':600,'margin':25,}
            ),

            html.Div([
                html.Table(id='table')
                ], style={'width':560, 'margin':50}
            ),
        ]),
])

@app.callback(
    Output(component_id='table', component_property='children'),
    [Input(component_id='movie-selector', component_property='value'),
    Input(component_id='year-slider', component_property='value'),
    Input(component_id='rating-slider', component_property='value')]
)
def update_table(movie_value,year,rating):

    if not movie_value:
        return html.P('Select at least one movie')

    else:

        rows = list(movie_value)

        agg_vector = np.mean(movie_factors[rows],axis=0)

        dotted = movie_factors.dot(agg_vector)
        matrix_norms = np.linalg.norm(movie_factors, axis=1)
        vector_norm = np.linalg.norm(agg_vector)
        matrix_vector_norms = np.multiply(matrix_norms, vector_norm)
        neighbors = np.divide(dotted, matrix_vector_norms)

        sim_score_df = pd.DataFrame({'matrix_row':np.arange(len(neighbors)),'sim_score':-neighbors})

        table_data = df.merge(sim_score_df,on='matrix_row')

        table_data = table_data[(~table_data.matrix_row.isin(rows))&\
        (table_data.year >= year[0])&(table_data.year <= year[1])&\
        (table_data.avg_rating >= rating[0])&(table_data.avg_rating <= rating[1])]\
        .sort_values('sim_score').reset_index()[['movie_name','avg_rating']]

    return make_dash_table(table_data.iloc[:20])


if __name__ == '__main__':
    app.run_server(debug=True)