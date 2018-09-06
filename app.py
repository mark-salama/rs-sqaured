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
    table = []
    for x,i in enumerate(df):
        html_row = [html.Td(x+1),html.Td(i)]
        # for i in range(len(row)):
        #     html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(children=[
    html.H1(children='Simple Movie Recommender'),

    dcc.Markdown("Select movies you love."
    ),

    dcc.Dropdown(
    id='movie_selector',

    options=[{'label': rows['movie_name'], 'value': rows['matrix_row']} for index,rows in df.iterrows()],

    multi=True),
    
    html.Br([]),
    html.Strong("Movie Recommendations"),

    html.Table(id='table'),

])

@app.callback(
    Output(component_id='table', component_property='children'),
    [Input(component_id='movie_selector', component_property='value')]
)
def update_table(movie_value):

    rows = list(movie_value)

    agg_vector = np.minimum.reduce(movie_factors[rows,:])

    dotted = movie_factors.dot(agg_vector)
    matrix_norms = np.linalg.norm(movie_factors, axis=1)
    vector_norm = np.linalg.norm(agg_vector)
    matrix_vector_norms = np.multiply(matrix_norms, vector_norm)
    neighbors = np.divide(dotted, matrix_vector_norms)

    return_cols = [x for x in np.argsort(-neighbors)[:20] if x not in rows]

    return make_dash_table(df[df.matrix_row.isin(return_cols)].movie_name)


if __name__ == '__main__':
    app.run_server(debug=True)