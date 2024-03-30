import os
import dotenv
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from cit_vis.parse import aggregate_by_group, get_grid_structure, get_edgelist

def load_data(trait, data_dir, df):
    groups = pd.read_pickle(os.path.join(data_dir, trait, "groups.pkl"))
    color_scale = create_color_scale(df[trait])
    grid = pd.read_pickle(os.path.join(data_dir, trait, "grid.pkl"))
    mice = pd.read_csv(os.path.join(data_dir, trait, "mice.csv"), index_col=0).filter(["Group ID"], axis=1)
    group_ids = [mice.loc[mouse_id, 'Group ID'] for mouse_id in df.index]
    return groups, grid, color_scale, group_ids


def create_color_scale(vector):
    min_val = min(vector)
    max_val = max(vector)
    # Use Matplotlib's built-in 'coolwarm' colormap
    cmap = plt.get_cmap('coolwarm')
    # Normalize the colormap to fit the range of the results
    norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    # Return both colormap and normalizer to apply on values
    return cmap, norm

def get_color(value, color_scale):
    cmap, norm = color_scale
    rgba_color = cmap(norm(value))
    # Convert RGBA to hexadecimal
    return mcolors.to_hex(rgba_color)

def create_sankey_diagram(groups, color_scale):
    sankey_df = get_edgelist(groups)
    print('sankey_df')
    print(sankey_df)
    labels = list(set(sankey_df['source'].values) | set(sankey_df['target'].values))
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels
        ),
        link=dict(
            source=[labels.index(x) for x in sankey_df['source']],
            target=[labels.index(x) for x in sankey_df['target']],
            value=sankey_df['value'],
            #label=sankey_df['edge_label']
        )
    )])
    return fig

def create_stripplot(df, trait, color_scale):
    # Assign a color to each group
    df['color'] = [get_color(t, color_scale) for t in df[trait]]
    fig = px.strip(df, x='Group ID', y=trait, color='color')
    fig.update_layout(template='plotly_white', showlegend=False, title='Trait scores by group')
    # Add a dark gray stroke to the strip plot points:
    fig.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGray')))
    # On hover, show the mouse ID and the trait score
    fig.update_traces(hovertemplate='Mouse ID: %{x}<br>Trait: %{y}')
    return fig

def create_grid_plot(feature, df):
    fig = px.histogram(df, x=feature, color='Group ID', color_discrete_sequence=['lightgrey', 'blue'], nbins=20, histnorm='probability density', barmode='overlay', title=f'{feature} abundance')
    # Add a vertical line to show the mean value
    fig.add_vline(x=df[feature].mean(), line_dash="dash", line_color="black")
    fig.update_layout(template='plotly_white', width=400, height=400)
    return fig

def grid_handler(grid, df):
    grid_dict = grid.to_dict(orient='index')
    plot_grid = []
    for col, cells in grid_dict.items():
        row_items = []
        for group, celldata in cells.items():
            if celldata['Type'] == 'Arrow':
                row_items.append(html.Div(style={'width': '400px', 'height': '400px'}))
            else:
                subset = df.loc[celldata['Mouse IDs']]
                row_items.append(html.Div([dcc.Graph(figure=create_grid_plot(celldata['Feature'], subset))]))
        row_items = html.Div(row_items, style={'display': 'flex', 'flex-direction': 'row', 'flex-wrap': 'wrap'})
        plot_grid.append(row_items)
    plot_grid = html.Div(plot_grid, style={'display': 'flex', 'flex-direction': 'column'})
    return plot_grid



def make_dashboard():
    dotenv.load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    trait_df = pd.read_csv(os.path.join(data_dir, "trait_scores.csv"), index_col=0)
    traits = trait_df.columns
    microbe_df = pd.read_csv(os.path.join(data_dir, "microbe_abundances.csv"), index_col=0)
    microbes = microbe_df.columns
    microbes_dropdown_options = [{'label': microbe, 'value': microbe} for microbe in microbes]
    microbes_dropdown_options.insert(0, {'label': 'All', 'value': 'All'})
    df = pd.concat([trait_df, microbe_df], axis=1)

    # Things that update when the microbe changes
    selected_microbe = 'All'
    trait_dropdown_options = [{'label': trait, 'value': trait} for trait in traits]
    selected_trait = traits[0]

    # Things that update when the trait changes
    groups, grid, color_scale, df['Group ID'] = load_data(selected_trait, data_dir, df)

    app = Dash(__name__)
    
    # Define the layout
    app.layout = html.Div([
        html.H1("CIT Visualization"),
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='microbe-dropdown',
                    options=microbes_dropdown_options,
                    value=selected_microbe
                )
            ]),
            html.Div([
                dcc.Dropdown(
                    id='trait-dropdown',
                    options=trait_dropdown_options,
                    value=selected_trait
                )
            ]),
        ]),
        html.Div([
            html.Div([
                dcc.Graph(id='sankey-diagram', figure=create_sankey_diagram(groups, color_scale))
            ], style={'width': '49%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='stripplot', figure=create_stripplot(df, selected_trait, color_scale))
            ], style={'width': '49%', 'display': 'inline-block'})
        ]),
        html.Div(id='grid-container', children=grid_handler(grid, df), style={'width': '100%'}),
        dcc.Store(id='selected-trait', data=selected_trait)
    ])
    @app.callback(
        [Output('trait-dropdown', 'options'),
         Output('trait-dropdown', 'value')],
        [Input('microbe-dropdown', 'value'),
         Input('selected-trait', 'data')]
    )
    def update_trait_dropdown(selected_microbe, selected_trait):
        if selected_microbe == 'All':
            return [{'label': trait, 'value': trait} for trait in traits], selected_trait
        relevant_traits = traits#get_relevant_traits(selected_microbe)
        if selected_trait not in relevant_traits:
            selected_trait = relevant_traits[0]
        return [{'label': trait, 'value': trait} for trait in relevant_traits], selected_trait
    
    @app.callback(
        [Output('sankey-diagram', 'figure'),
         Output('stripplot', 'figure'),
         Output('grid-container', 'children')],
        [Input('trait-dropdown', 'value')]
    )
    def update_charts(selected_trait):
        groups, grid, color_scale, df['Group ID'] = load_data(selected_trait, data_dir, df)
        sankey = create_sankey_diagram(groups, color_scale)
        stripplot = create_stripplot(df, selected_trait, color_scale)
        grid = grid_handler(grid, df)
        return sankey, stripplot, grid

    app.run_server(debug=True)
    
if __name__ == "__main__":
    make_dashboard()