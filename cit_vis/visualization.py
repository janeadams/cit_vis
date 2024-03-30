import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from cit_vis.parse import get_trait_scores, get_microbe_abundances, group_by_groups, get_relevant_traits, get_mice, create_matrix

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

def get_mean_value(df, group):
    group_df = group_by_groups(df)
    mean_val = group_df.loc[group]['Mean']
    return mean_val

def create_sankey_df(df):
    sankey_df = pd.DataFrame(columns=['source', 'target', 'value'])
    for i, row in df.iterrows():
        path = row['Path'].split('>')
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            if source not in sankey_df['source'].values:
                sankey_df = pd.concat([sankey_df, pd.DataFrame({'source': source, 'target': target, 'value': [1]})])
            else:
                if target in sankey_df[sankey_df['source'] == source]['target'].values:
                    sankey_df.loc[(sankey_df['source'] == source) & (sankey_df['target'] == target), 'value'] += 1
                else:
                    sankey_df = pd.concat([sankey_df, pd.DataFrame({'source': source, 'target': target, 'value': [1]})])
    # Create informative labels for the groups
    sankey_df['edge_label'] = sankey_df['source'] + ' to ' + sankey_df['target']
    return sankey_df

def create_sankey_diagram(df, color_scale):
    sankey_df = create_sankey_df(df)
    labels = list(set(sankey_df['source'].values) | set(sankey_df['target'].values))
    color = [get_color(get_mean_value(df, group), color_scale) if 'Group' in group else 'black' for group in labels]
    fig = go.Figure(data=[go.Sankey(
        group=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=labels,
            color=color
        ),
        link=dict(
            source=[labels.index(x) for x in sankey_df['source']],
            target=[labels.index(x) for x in sankey_df['target']],
            value=sankey_df['value'],
            label=sankey_df['edge_label']
        )
    )])
    return fig

def create_stripplot(df, color_scale):
    # Sort by the mean group value
    group_df = group_by_groups(df)
    group_df = group_df.sort_values(by='Mean')
    group_order = group_df.index
    df = df.set_index('Group ID').loc[group_order].reset_index()
    # Assign a color to each group group
    df['color'] = [get_color(value, color_scale) for value in df['Trait']]
    color_lookup = dict(zip(df['Group ID'], df['color']))
    fig = px.strip(df, x='Group ID', y='Trait', color='Group ID', color_discrete_map=color_lookup)
    fig.update_layout(template='plotly_white', showlegend=False, title='Trait scores by group')
    # Add a dark gray stroke to the strip plot points:
    fig.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGray')))
    # On hover, show the mouse ID and the trait score
    fig.update_traces(hovertemplate='Mouse ID: %{x}<br>Trait: %{y}')
    return fig

def create_grid_plot(trait, group, microbe):
    mice = get_mice(trait, group)
    df = get_microbe_abundances()
    df['group'] = [group if x else 'other' for x in df.index.isin(mice)]
    # Create a histogram with the microbe abundance
    fig = px.histogram(df, x=microbe, color='group', color_discrete_sequence=['lightgrey', 'blue'], nbins=20, histnorm='probability density', barmode='overlay', title=f'{microbe} abundance in group {group}')
    # Add a vertical line to show the mean value
    fig.add_vline(x=df[microbe].mean(), line_dash="dash", line_color="black")
    fig.update_layout(template='plotly_white', width=400, height=400)
    return fig

def grid_handler(df, trait):
    #matrix = create_matrix(df, trait)
    grid_data = []
    for microbe in matrix.columns:
        row_items = [html.Div(f'{microbe}')]
        for group in matrix.index:
            if matrix.loc[group, microbe] == 0:
                row_items.append(html.Div(style={'width': '400px', 'height': '400px'}))
            else:
                row_items.append(html.Div([dcc.Graph(figure=create_grid_plot(trait, group, microbe))]))
        row_items = html.Div(row_items, style={'display': 'flex', 'flex-direction': 'row', 'flex-wrap': 'wrap'})
        plot_grid.append(row_items)
    plot_grid = html.Div(plot_grid, style={'display': 'flex', 'flex-direction': 'column'})
    return plot_grid

def make_dashboard():
    trait_df = get_trait_scores()
    traits = trait_df.columns
    microbe_df = get_microbe_abundances()
    microbes = microbe_df.columns
    microbes_dropdown_options = [{'label': microbe, 'value': microbe} for microbe in microbes]
    microbes_dropdown_options.insert(0, {'label': 'All', 'value': 'All'})
    df = pd.concat([trait_df, microbe_df], axis=1)

    # Things that update when the microbe changes
    selected_microbe = 'All'
    trait_dropdown_options = [{'label': trait, 'value': trait} for trait in traits]
    selected_trait = traits[0]

    # Things that update when the trait changes
    color_scale = create_color_scale(df[selected_trait])

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
                dcc.Graph(id='sankey-diagram', figure=create_sankey_diagram(df, color_scale))
            ], style={'width': '49%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='stripplot', figure=create_stripplot(df, color_scale))
            ], style={'width': '49%', 'display': 'inline-block'})
        ]),
        html.Div(id='grid-container', children=grid_handler(df, selected_trait), style={'width': '100%'}),
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
        relevant_traits = get_relevant_traits(selected_microbe)
        if selected_trait not in relevant_traits:
            selected_trait = relevant_traits[0]
        return [{'label': trait, 'value': trait} for trait in relevant_traits], selected_trait

    @app.callback(
        [Output('sankey-diagram', 'figure'),
         Output('stripplot', 'figure')],
        [Input('trait-dropdown', 'value')]
    )
    def update_sankey_diagram(selected_trait):
        color_scale = create_color_scale(df[selected_trait])
        return create_sankey_diagram(df, color_scale), create_stripplot(df, color_scale)

    @app.callback(
        Output('grid-container', 'children'),
        [Input('trait-dropdown', 'value')]
    )
    def update_grid(selected_trait):
        return grid_handler(df, selected_trait, microbes, groups)

    app.run_server(debug=True)