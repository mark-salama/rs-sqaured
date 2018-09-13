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
    html_row = [html.Th('Rank'),html.Th('Movie'),html.Th('Rating')]
    table = []
    table.append(html.Tr(html_row))

    for index,row in df.iterrows():
        html_row = [html.Td(index+1),html.Td(row.movie_name),html.Td(round(row.avg_rating,2))]
        table.append(html.Tr(html_row))
    return table

app = dash.Dash()

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Movie Recommender</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        <div></div>
        {%app_entry%}
        <footer>
        
        Movie features were learned using the KNN similarity function provided by <a href="http://surpriselib.com">Surprise</a>. Ratings data from the <a href="https://grouplens.org/datasets/movielens/">MovieLens dataset</a>, citation below. 

        <br>

        F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. DOI=http://dx.doi.org/10.1145/2827872

            {%config%}
            {%scripts%}
        </footer>
    </body>
</html>
'''


app.layout = html.Div(children=[

    html.Div([  

        html.H2("""Find Movies You'll Love""",
            style={
                'text-align': 'center' ,
                'margin-left': 'auto',
                'margin-right': 'auto' 

            }
        ),

        html.Div([  

            dcc.Dropdown(
                id='movie-selector',
                options=[{'label': rows['movie_name'], 'value': rows['matrix_row']} for index,rows in df.sort_values('avg_rating',ascending=False).iterrows()],
                multi=True),
    
        ],
        style={'width': '75%', 'display': 'inline-block', 'marginBottom':15}
        ),

        html.Br([]),

        html.H6('Year',
            style={
                'display':'inline-block',
            }
        ),

        html.Div([
            dcc.RangeSlider(
                id='year-slider',
                min=1940,
                max=2020,
                step=5,
                marks={
                    1940: '1940',
                    1960: '1960',
                    1980: '1980',
                    2000: '2000',
                    2020: '2020',
                    },
                value=[1970, 2020]
            ),
        ],
        style={'width':'50%', 'display':'inline-block','margin-left':28,'margin-right': 'auto'}
        ),

        html.Br([]),
        html.Br([]),
        
        html.H6('Rating',
            style={
                'display':'inline-block',
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
                value=[2, 5]
            )
        ], 
        style={'width':'50%', 'display':'inline-block','margin-left':20,'margin-right': 'auto'  }
        ),

        html.Br([]),
        html.Br([]),

        html.H6('Results',
            style={
                'display':'inline-block',
            }
        ),

        html.Div([
            dcc.Slider(
                id='output-count-slider',
                min=20,
                max=100,
                step=20,
                marks={
                    20: '20',
                    40: '40',
                    60: '60',
                    80: '80',
                    100: '100'
                    },
                value=20
            )
        ], 
        style={'width':'50%', 'display':'inline-block', 'margin-left':20 }

        ),

        html.Br([]),
        html.Br([]),

        html.Table(id='table')


    ],
    style={'margin':25, 'text-align': 'center' ,}
    ),
])

@app.callback(
    Output(component_id='table', component_property='children'),
    [Input(component_id='movie-selector', component_property='value'),
    Input(component_id='year-slider', component_property='value'),
    Input(component_id='rating-slider', component_property='value'),
    Input(component_id='output-count-slider', component_property='value')]
)
def update_table(movie_value,year,rating,output):

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

    return make_dash_table(table_data.iloc[:output])


if __name__ == '__main__':
    app.run_server(debug=True)