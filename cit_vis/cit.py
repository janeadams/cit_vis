import os
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from scipy import interpolate
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import pickle
import networkx as nx


def find_traits(microbe):
    """
    Given a summary tree, find traits for a specific microbe and return as an array.

    Parameters:
        microbe (str): The microbe name for which traits need to be found.

    Returns:
        list: A list containing unique traits associated with the specified microbe.
    """
    # Load the processed summary tree from the pickle file.
    summary_tree = pd.read_pickle('new_processed_tree.pkl')

    # Filter the summary tree based on the specified microbe.
    filt_source = summary_tree[summary_tree['microbe'] == microbe]

    # Extract the unique traits for the specified microbe and store them in a list.
    b_list = list(set(filt_source['trait']))

    return b_list

def find_trees(path='data/conditional_inference_tree_results'):
    """
    Traverse the file structure to find all conditional inference trees;
    Return a list of traits, trees, and their paths.

    Returns:
        tuple: A tuple containing three lists:
            1. A list of traits (directory names)
            2. A list of conditional inference tree filenames
            3. A list of absolute paths to the corresponding trees
    """
    # Initialize empty lists to store traits, trees, and their paths.
    traits = []
    trees = []
    tree_paths = []

    # Traverse the file structure starting from the specified directory.
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            # Check if the file matches the criteria to be a conditional inference tree file.
            if ('TreeSplits' in name) & ('ranknorm' in name) & ('.txt' in name) & ('raw' not in name):
                # If the file meets the criteria, add its name to the trees list and its absolute path to the tree_paths list.
                trees.append(name)
                tree_paths.append(os.path.join(root, name))

        for name in dirs:
            # Add the directory name to the traits list.
            traits.append(name)

    # Save the traits list as a pickle file named 'traits.pkl'.
    with open('traits.pkl', 'wb') as f:
        pickle.dump(traits, f)

    # Return the three lists as a tuple.
    return traits, trees, tree_paths

def read_abundance():
    """
    Read abundance data from a CSV file and transpose it to get a DataFrame with microbes as rows and samples as columns.

    Returns:
        tuple: A tuple containing two objects:
            1. A DataFrame with abundance data, where rows represent microbes and columns represent samples.
            2. A pandas Index object containing the names of microbes.
    """
    # Read abundance data from the CSV file and transpose it.
    all_abundance = pd.read_csv('data/cecum_microbe_counts_rankZ.csv', index_col=[0]).T

    # Get the names of the microbes from the columns of the DataFrame.
    microbes = all_abundance.columns

    # Return the DataFrame and the pandas Index object containing microbes names as a tuple.
    return all_abundance, microbes


def read_trait():
    """
    Read trait data from a CSV file and transpose it to get a DataFrame with traits as rows and samples as columns.

    Returns:
        tuple: A tuple containing two objects:
            1. A DataFrame with trait data, where rows represent traits and columns represent samples.
            2. A pandas Index object containing the names of traits.
    """
    # Read trait data from the CSV file and transpose it.
    all_trait = pd.read_csv('data/novelty_traits_rankZ.csv', index_col=[0]).T

    # Get the names of the traits from the columns of the DataFrame.
    traits = all_trait.columns

    # Return the DataFrame and the pandas Index object containing trait names as a tuple.
    return all_trait, traits


def get_split_df(trait):
    """
    Given a trait, traverse the file structure to find and process group CSV files.
    Construct a DataFrame (split_df) containing group information and a list of DataFrames (group_dfs) for each group.

    Parameters:
        trait (str): The trait for which to retrieve group information.

    Returns:
        tuple: A tuple containing two objects:
            1. A DataFrame (split_df) with information from the group CSV file corresponding to 'group.split.values'.
            2. A list of DataFrames (group_dfs) containing information from each group CSV file (excluding 'group.split.values').
    """
    group_paths = []  # List to store the paths of group CSV files.
    group_dfs = []    # List to store DataFrames of each group CSV file.

    split_df = pd.DataFrame()  # DataFrame to store information from the 'group.split.values' CSV file.

    # Traverse the file structure for the specified trait.
    for root, dirs, files in os.walk(f'conditional_inference_tree_results/{trait}', topdown=False):
        for name in files:
            if ('group' in name) & ('.csv' in name):
                # If the file name contains 'group' and ends with '.csv', it's a group CSV file.
                group_path = os.path.join(root, name)
                group_paths.append(group_path)

                group_df = pd.read_csv(group_path)  # Read the group CSV file.

                group_number = name.split('_')[-1][4:-4]  # Extract the group number from the file name.

                if group_number != 'group.split.values':
                    # If it's not the 'group.split.values' file, process and add it to the group_dfs list.
                    group_df['group'] = group_number
                    group_df['group_avg'] = group_df[trait].mean()

                    # Find the column containing the split_microbe and assign it to the 'split_microbe' variable.
                    for c in group_df.columns:
                        if ('ranknorm' in c) & ('g__' in c):
                            split_microbe = c

                    group_df['split_microbe'] = split_microbe
                    group_df['microbe_val'] = group_df[split_microbe]
                    group_dfs.append(group_df)
                else:
                    # If it's the 'group.split.values' file, assign it to the split_df DataFrame.
                    split_df = group_df

    return split_df, group_dfs


def map_ranks(cols):
    """
    Map rank values to descriptive trait names based on their position in the input list.

    Parameters:
        cols (list): A list containing rank values (trait levels) to be mapped to descriptive trait names.

    Returns:
        dict: A dictionary mapping rank values to corresponding descriptive trait names.
    """
    new_names = {}  # Create an empty dictionary to store the mappings.

    for i, col in enumerate(cols):
        if i == 0:
            name = 'Very low trait'
        elif i == 1:
            name = 'Low trait'
        # elif i == 2:
        #     name = 'Somewhat low trait'
        elif i == (len(cols) - 1):
            name = 'Very high trait'
        elif i == (len(cols) - 2):
            name = 'High trait'
        # elif i == (len(cols) - 3):
        #     name = 'Somewhat high trait'
        else:
            name = 'medium'

        new_names[col] = name  # Add the mapping of the rank value to the trait name in the dictionary.

    return new_names


def get_leaf_df(trait, grouped=False):
    """
    Given a trait, retrieve leaf information related to that trait.

    Parameters:
        trait (str): The trait for which to retrieve leaf information.
        grouped (bool, optional): If True, the leaf groups will be grouped into 'low', 'medium', and 'high' based on their rank. If False, groups will be grouped based on their numeric rank. Defaults to False.

    Returns:
        pandas.DataFrame: A DataFrame containing leaf information, including Mouse_ID, trait values, group information, group averages, split_microbe, microbe_val, group_rank, and group.
    """
    # Retrieve split_df and group_dfs using the get_split_df function.
    split_df, group_dfs = get_split_df(trait)

    # Extract the leaves from split_df.
    leaves = list(split_df[split_df['inner_group_pvalue'].isnull()]['group'].values)

    # Concatenate the group DataFrames and sort by 'group_avg'.
    leaf_df = pd.concat(group_dfs).sort_values(by='group_avg')

    # Filter leaf_df to include only rows corresponding to leaf groups.
    leaf_df = leaf_df[leaf_df['group'].isin([str(l) for l in leaves])]

    # Select relevant columns for the leaf_df.
    leaf_df = leaf_df.filter(['Mouse_ID', trait, 'group', 'group_avg', 'split_microbe', 'microbe_val'])

    # Create a dictionary for mapping group values to their ranks.
    n_ordlist = list(dict.fromkeys(leaf_df['group']))
    group_rank = dict(zip(n_ordlist, range(0, len(n_ordlist))))

    # Add a new column 'group_rank' to leaf_df, which contains the rank of each group.
    leaf_df['group_rank'] = [group_rank[n] for n in leaf_df['group']]

    if grouped:
        # If 'grouped' is True, group the leaf groups into 'low', 'medium', and 'high' based on their rank.
        leaf_df['group'] = ['low' if n in n_ordlist[:2] else 'high' if n in n_ordlist[-2:] else 'medium' for n in leaf_df['group']]
    else:
        # If 'grouped' is False, group the leaf groups based on their numeric rank.
        rank_map = map_ranks(list(set(leaf_df['group_rank'])))
        leaf_df['group'] = [rank_map[n] for n in leaf_df['group_rank']]

    # Duplicate Check
    leaf_df['isDupe'] = leaf_df.duplicated(subset=['Mouse_ID'], keep=False)
    leaf_df[leaf_df['isDupe'] == True].sort_values(by='Mouse_ID')

    # Read abundance data and add it to the leaf_df for each microbe.
    all_abundance = read_abundance()[0]
    for microbe in list(all_abundance.columns):
        if microbe+'_ranknorm' in list(set(leaf_df['split_microbe'])):
            vals = []
            for mouse in leaf_df['Mouse_ID']:
                vals.append(all_abundance.loc[[str(mouse)]][microbe][0])
            leaf_df[microbe] = vals

    return leaf_df



def assign_random_colors(cols):
    """
    Assign random colors to each unique value in the input list 'cols' and create a dictionary to map them.

    Parameters:
        cols (list): A list of unique values for which random colors are to be assigned.

    Returns:
        dict: A dictionary mapping each unique value in 'cols' to a corresponding random color.
    """
    # Predefined color options from Plotly's qualitative palette 'Bold'.
    options = px.colors.qualitative.Bold

    # Dictionary containing predefined color assignments for specific values.
    assigned = {
        'high': 'red',
        'medium': 'grey',
        'low': 'blue',
        'all': 'LightGray',
    }

    i = 0
    for col in cols:
        if i >= len(options):
            # If all predefined colors are used, reset the index back to zero.
            i = 0

        if col not in assigned.keys():
            # If the value is not already assigned a color, assign the next available color from options.
            assigned[col] = options[i]
            i += 1

    return assigned


def assign_diverging_colors(cols):
    """
    Assign specific diverging colors to elements in the input list 'cols' and create a dictionary to map them.

    Parameters:
        cols (list): A list of elements for which specific diverging colors are to be assigned.

    Returns:
        dict: A dictionary mapping each element in 'cols' to its corresponding diverging color.
    """
    assigned = {}  # Initialize an empty dictionary for color assignments.

    try:
        # Try to assign specific colors to elements at specific positions in 'cols' if they exist.
        # This is done using index-based color assignments.
        assigned[cols[-3]] = 'yellowgreen'
        assigned[cols[2]] = 'Yellow'
    except:
        pass

    # Update the color assignments using specific colors for elements at certain positions.
    assigned = {
        cols[1]: 'Orange',
        cols[-2]: 'yellowgreen',
        cols[0]: 'darkred',
        cols[-1]: 'darkgreen'
    }

    # Assign specific colors to the values 'medium' and 'all'.
    assigned['medium'] = 'lightgrey'
    assigned['all'] = '#eee'

    return assigned

def show_strip(b):
    """
    Create and display a strip plot using Plotly.

    Parameters:
        b (str): The trait for which the strip plot needs to be created.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object representing the strip plot.
    """
    # Get the leaf information DataFrame related to the specified trait.
    leaf_df = get_leaf_df(b)

    # Create a strip plot using Plotly express, setting 'y' as the specified trait,
    # 'color' as the 'group' column from the leaf_df DataFrame, and using specific colors from the assign_diverging_colors function.
    fig = px.strip(leaf_df, y=b, color="group", color_discrete_map=assign_diverging_colors(list(leaf_df['group'].unique())), hover_data=['Mouse_ID'])

    # Update the layout of the figure to use a white template, and set the height and width.
    fig.update_layout(template='plotly_white', height=300, width=500)

    return fig


def categorize_abundance(microbe, all_abundance=read_abundance()[0]):
    """
    Categorize the abundance of a microbe for each mouse into 'low', 'medium', or 'high' groups.

    Parameters:
        microbe (str): The microbe for which abundance needs to be categorized.
        all_abundance (pandas.DataFrame, optional): The DataFrame containing abundance data for all microbes. Defaults to the abundance data obtained from read_abundance().

    Returns:
        pandas.DataFrame: A DataFrame containing the abundance values and the assigned group for each mouse.
    """
    # Filter mice with low and high abundance based on specified thresholds (< -1 and > 1, respectively).
    low_mice = list(all_abundance[all_abundance[microbe] < -1].index)
    high_mice = list(all_abundance[all_abundance[microbe] > 1].index)

    # Create a new DataFrame containing the abundance of the specified microbe and the assigned group for each mouse.
    new_df = all_abundance.copy()
    new_df = new_df.filter([microbe])
    new_df['group'] = ['low' if mouse in low_mice else 'high' if mouse in high_mice else 'medium' for mouse in list(all_abundance.index)]
    new_df = new_df.sort_values(by='group')

    return new_df


def get_abundance_df(microbe, all_trait=read_trait()[0]):
    """
    Combine abundance data and trait data for a given microbe and create a DataFrame with mouse abundance and trait information.

    Parameters:
        microbe (str): The microbe for which abundance and trait data needs to be combined.
        all_trait (pandas.DataFrame, optional): The DataFrame containing trait data for all microbes. Defaults to the trait data obtained from read_trait().

    Returns:
        pandas.DataFrame: A DataFrame containing mouse ID, abundance, and trait information for the specified microbe.
    """
    # Categorize the abundance of the specified microbe for each mouse into 'low', 'medium', or 'high' groups.
    ab_cats = categorize_abundance(microbe)

    # Filter the trait data to include only the traits associated with the specified microbe.
    trait_subset = all_trait.filter(find_traits(microbe))

    # Combine the abundance data and trait data using the mouse ID as the index.
    abundance_df = ab_cats.join(trait_subset)

    # Reset the index and rename the column 'index' to 'Mouse_ID'.
    abundance_df = abundance_df.reset_index().rename(columns={'index': 'Mouse_ID'})

    return abundance_df

def get_trait_df(trait, summary_tree=None, abundance_data=read_abundance()[0]):
    """
    Get a DataFrame containing abundance data for the microbes associated with the specified trait.

    Parameters:
        trait (str): The trait for which abundance data is needed.
        summary_tree (pandas.DataFrame, optional): The DataFrame containing the summary tree. Defaults to the DataFrame obtained from reading 'new_processed_tree.pkl'.
        abundance_data (pandas.DataFrame, optional): The DataFrame containing abundance data for all microbes. Defaults to the abundance data obtained from read_abundance().

    Returns:
        pandas.DataFrame: A DataFrame containing abundance data for the microbes associated with the specified trait.
    """
    if summary_tree is None:
        try:
            summary_tree = pd.read_pickle('new_processed_tree.pkl')
        except:
            print('Did you run find_trees() yet?')

    # Filter the summary tree to get a subset containing only the rows with the specified trait.
    subset = summary_tree[summary_tree['trait'] == trait]

    # Get the unique microbes associated with the specified trait.
    microbes = list(set(subset['microbe']))

    # Filter the abundance data to include only the columns corresponding to the microbes associated with the trait.
    subdf = abundance_data.filter(microbes)

    return subdf


def get_microbes(cols):
    """
    Extract column names that contain the string 'g__' in them.

    Parameters:
        cols (list): A list of column names.

    Returns:
        list: A new list containing only the column names that contain 'g__' in them.
    """
    g_list = []
    for c in cols:
        if 'g__' in c:
            g_list.append(c)

    return g_list


def get_traits(cols):
    """
    Extract column names that contain the string 'ranknorm' in them.

    Parameters:
        cols (list): A list of column names.

    Returns:
        list: A new list containing only the column names that contain 'ranknorm' in them.
    """
    b_list = []
    for c in cols:
        if 'ranknorm' in c:
            b_list.append(c)

    return b_list


def clean(term):
    """
    Clean and modify the format of a term (string).

    Parameters:
        term (str): The term to be cleaned.

    Returns:
        str: The cleaned and modified term.
    """
    for s in ['g__', '_batch_ranknorm']:
        # Remove 'g__' and '_batch_ranknorm' from the term by replacing them with an empty string.
        term = term.replace(s, '')

    # Replace underscores '_' with spaces ' ' in the term.
    term = term.replace('_', ' ')

    return term


def tuple_titles(cols, rows):
    """
    Create a tuple of formatted column titles and empty strings to align with rows in a tabular format.

    Parameters:
        cols (list): A list of column names.
        rows (list): A list of row names.

    Returns:
        tuple: A tuple containing formatted column titles and empty strings to align with rows.
    """
    # Capitalize each column title and add it to a new list 'title_formatted'.
    title_formatted = [f'{col.title()}' for col in cols]

    # Create a new tuple 'title_tuple' by concatenating 'title_formatted' list with empty strings.
    # The number of empty strings is determined by the difference between the number of rows and the number of columns.
    title_tuple = tuple(title_formatted + [' '*(len(rows)-1)])

    return title_tuple


def setup_chart(subject, grouped=False, drop='medium', diverging=True):
    """
    Set up data and configurations for a chart.

    Parameters:
        subject (str): The subject of the chart, either 'microbe' or a specific trait.
        grouped (bool, optional): If True, the data will be grouped. Defaults to False.
        drop (str, optional): A category to be dropped from the 'cols'. Defaults to 'medium'.
        diverging (bool, optional): If True, diverging colors will be used. Defaults to True.

    Returns:
        tuple: A tuple containing view_df (DataFrame), cols (list), rows (list), and color_lookup (dict).
    """
    if 'g__' in subject:
        view = 'microbe'
        view_df = get_abundance_df(subject)
        rows = get_traits(view_df.columns)
    else:
        view = 'trait'
        view_df = get_leaf_df(subject, grouped=grouped)
        rows = get_microbes(view_df.columns)

    cols = list(view_df['group'].unique())
    try:
        cols.remove(drop)
    except:
        print(f'{drop} not in {cols}')

    if diverging:
        color_lookup = assign_diverging_colors(cols)
    else:
        color_lookup = assign_random_colors(cols)

    return view_df, cols, rows, color_lookup



def make_violin_matrix(subject, grouped=False, drop='medium'):
    """
    Generate a matrix of violin plots based on the specified subject (microbe or trait).

    Parameters:
        subject (str): The subject for which the matrix of violin plots needs to be created.
        grouped (bool, optional): If True, the data will be grouped. Defaults to False.
        drop (str, optional): A category to be dropped from the 'cols'. Defaults to 'medium'.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object representing the matrix of violin plots.
    """
    # Set up data and configurations for the chart using the setup_chart function.
    view_df, cols, rows, color_lookup = setup_chart(subject, grouped=grouped, drop=drop)

    # Create a matrix of violin plots using Plotly's make_subplots function.
    fig = make_subplots(rows=len(rows), cols=len(cols), subplot_titles=tuple_titles(cols, rows), horizontal_spacing=0.05, vertical_spacing=0.05)

    for i, cohort in enumerate(cols):
        for j, row in enumerate(rows):

            # Create a title for each subplot indicating the group and cohort.
            title = f'Group {cohort.title()}'

            # Subset the data for the specific cohort and row.
            subset = view_df[view_df['group'] == str(cohort)]

            # Add a Violin plot to the figure for the specific cohort and row.
            # The Violin plot represents the distribution of data for the given cohort and row.
            # Additionally, a Violin plot for 'all' rows is added to show the overall distribution.
            fig.add_trace(
                go.Violin(
                    y=subset[row],
                    side='negative',
                    legendgroup=cohort,
                    offsetgroup=row,
                    name=cohort,
                    line_color=color_lookup[cohort],
                    showlegend=False,
                    x0=f'{clean(row)} <br>for cohort<br>{cohort}',
                    pointpos=-1.5,
                    hovertext=[f'Mouse ID: {n}' for n in subset["Mouse_ID"]],
                    hoverinfo="text",
                    points='all'
                ),
                row=j+1,
                col=i+1
            )
            fig.add_trace(
                go.Violin(
                    y=view_df[row],
                    side='positive',
                    legendgroup='all',
                    offsetgroup=row,
                    name='all',
                    showlegend=False,
                    line_color='lightgray',
                    x0=f'{clean(row)} <br>for cohort<br>{cohort}',
                    pointpos=1.5,
                    hovertext=[f'Mouse ID: {n}' for n in view_df["Mouse_ID"]],
                    hoverinfo="text",
                ),
                row=j+1,
                col=i+1
            )

            # Update the y-axis range to ensure that all plots have the same y-axis scale.
            fig.update_yaxes(range=[view_df[row].min()-1, view_df[row].max()+1])

            # If it is the first column, update the y-axis title to display the row name.
            if i == 0:
                fig.update_yaxes(title_text=f'{clean(row).replace(" ", "<br>")}', row=j+1, col=i+1)

    # Update the plot's traces and layout.
    fig.update_traces(meanline_visible=True, width=0.4, points='all')
    fig.update_layout(violingap=0, violinmode='overlay', height=250*len(rows), template='plotly_white', title=clean(subject))

    return fig

def make_simple_scatter(trait, microbe):
    """
    Generate a scatter plot based on the specified trait and microbe data.

    Parameters:
        trait (str): The trait for the x-axis of the scatter plot.
        microbe (str): The microbe for the y-axis of the scatter plot.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object representing the scatter plot.
    """
    # Get the DataFrame containing the leaf data for the specified trait.
    df = get_leaf_df(trait)

    # Assign diverging colors to different groups in the scatter plot.
    color_lookup = assign_diverging_colors(list(set(df['group'])))

    # Create the scatter plot using Plotly's px.scatter function.
    fig = px.scatter(df, x=trait, y=microbe, color='group', color_discrete_map=color_lookup)

    # Update the appearance of the scatter plot's markers.
    fig.update_traces(marker=dict(size=12, opacity=0.8,
                              line=dict(width=1,
                                        color='DarkSlateGrey')))

    # Update the layout of the scatter plot.
    fig.update_layout(template='plotly_white')

    return fig

def make_scatter_column(trait):
    """
    Generate a column of scatter plots based on the specified trait.

    Parameters:
        trait (str): The trait for the x-axis of the scatter plots.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object representing the column of scatter plots.
    """
    # Get the DataFrame containing the leaf data for the specified trait.
    leaf_df = get_leaf_df(trait)

    # Get the microbe names present in the leaf DataFrame.
    gen_names = []
    for col in leaf_df.columns:
        if 'g__' in col:
            gen_names.append(col)

    # Assign diverging colors to different groups in the scatter plots.
    color_lookup = assign_diverging_colors(list(leaf_df['group'].unique()))

    # Create the column layout using Plotly's make_subplots function.
    fig = make_subplots(rows=len(gen_names), cols=1, vertical_spacing=0.05)

    for i, gen in enumerate(gen_names):
        for j, cohort in enumerate(list(set(leaf_df['group']))):
            subset = leaf_df[leaf_df['group'] == cohort]

            # Add a Scatter plot to the figure for the specific microbe and cohort.
            # Each subplot in the column represents a different microbe (y-axis) against the trait (x-axis).
            fig.add_trace(
                go.Scatter(
                    x=subset[trait],
                    y=subset[gen],
                    legendgroup=cohort,
                    name=cohort,
                    line_color=color_lookup[cohort],
                    mode='markers',
                    showlegend=True if i == 0 else False,
                    x0=f'{gen}',
                    hovertext=[f'Mouse ID: {n}' for n in leaf_df["Mouse_ID"]],
                    hoverinfo="text",
                ),
                row=i+1,
                col=1
            )

            # Update y-axis title to display the microbe name.
            fig.update_yaxes(title_text=f'{gen[3:].replace("_", " ")}', row=i+1, col=1)

            # Update x-axis title to display the trait name.
            fig.update_xaxes(title_text=f'{trait[:-15].replace("_", " ")}', row=i+1, col=1)

    # Update the appearance of the scatter plot's markers.
    fig.update_traces(marker=dict(size=10, opacity=0.5,
                              line=dict(width=1,
                                        color='DarkSlateGrey')))

    # Update the layout of the figure.
    fig.update_layout(template='plotly_white', title=clean(trait), height=250*len(gen_names))

    return fig

def make_scatter_matrix(subject, grouped=False, drop='medium'):
    """
    Generate a matrix of scatter plots based on the specified subject (microbe or trait).

    Parameters:
        subject (str): The subject for which the matrix of scatter plots needs to be created.
        grouped (bool, optional): If True, the data will be grouped. Defaults to False.
        drop (str, optional): A category to be dropped from the 'cols'. Defaults to 'medium'.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object representing the matrix of scatter plots.
    """
    # Set up data and configurations for the chart using the setup_chart function.
    view_df, cols, rows, color_lookup = setup_chart(subject, grouped=grouped, drop=drop)

    # Create a matrix of scatter plots using Plotly's make_subplots function.
    fig = make_subplots(rows=len(rows), cols=len(cols), subplot_titles=tuple_titles(cols, rows), horizontal_spacing=0.05, vertical_spacing=0.05)

    for i, cohort in enumerate(cols):
        for j, row in enumerate(rows):

            # Create a title for each subplot indicating the group and cohort.
            title = f'Group {cohort.title()}'

            # Subset the data for the specific cohort and row.
            subset = view_df[view_df['group'] == str(cohort)]

            # Add a Scatter plot to the figure for the overall distribution (all rows).
            # The Scatter plot represents the distribution of data for the given cohort and the specified subject (x-axis).
            fig.add_trace(
                go.Scatter(
                    x=view_df[subject],
                    y=view_df[row],
                    mode='markers',
                    legendgroup='all',
                    name='all',
                    showlegend=False,
                    line_color='#eee',
                    opacity=0.5,
                    x0=f'{clean(row)} <br>for cohort<br>{cohort}',
                    hovertext=[f'Mouse ID: {n}' for n in view_df["Mouse_ID"]],
                    hoverinfo="text"
                ),
                row=j+1,
                col=i+1
            )

            # Add a Scatter plot to the figure for the specific cohort and row.
            # This Scatter plot represents the distribution of data for the given cohort and row.
            fig.add_trace(
                go.Scatter(
                    x=subset[subject],
                    y=subset[row],
                    mode='markers',
                    x0=f'{clean(row)} <br>for cohort<br>{cohort}',
                    legendgroup=cohort,
                    name=cohort,
                    line_color=color_lookup[cohort],
                    hovertext=[f'Mouse ID: {n}' for n in subset["Mouse_ID"]],
                    hoverinfo="text"
                ),
                row=j+1,
                col=i+1
            )

            # If it is the first column, update the y-axis title to display the row name.
            if i == 0:
                fig.update_yaxes(title_text=f'{clean(row).replace(" ", "<br>")}', row=j+1, col=i+1)

    # Update the layout of the figure.
    fig.update_layout(violingap=0, violinmode='overlay', height=250*len(rows), template='plotly_white', title=clean(subject))

    return fig



def add_all(view_df, drop='medium'):
    """
    Combine the original DataFrame with a new DataFrame that contains the data for all groups combined.

    Parameters:
        view_df (pd.DataFrame): The original DataFrame containing the data for different groups.
        drop (str, optional): A category to be dropped from the original DataFrame. Defaults to 'medium'.

    Returns:
        pd.DataFrame: A new DataFrame that contains the combined data for all groups.
    """
    # If the 'drop' category is present in the DataFrame, remove rows with that category.
    # Otherwise, create a copy of the DataFrame.
    if drop in list(set(view_df['group'])):
        df = view_df[view_df['group'] != drop]
    else:
        df = view_df.copy()

    # Create a new DataFrame containing the data for all groups combined.
    all_df = view_df.copy()
    all_df['group'] = 'all'

    # Concatenate the original DataFrame (without the 'drop' category) and the 'all' DataFrame.
    combined = pd.concat([df, all_df])

    return combined

def make_strip(subject, grouped=False, drop='medium', diverging=False):
    """
    Generate a strip plot based on the specified subject (microbe or trait).

    Parameters:
        subject (str): The subject for which the strip plot needs to be created.
        grouped (bool, optional): If True, the data will be grouped. Defaults to False.
        drop (str, optional): A category to be dropped from the 'cols'. Defaults to 'medium'.
        diverging (bool, optional): If True, the colors will be diverging. Defaults to False.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object representing the strip plot.
    """
    # Set up data and configurations for the chart using the setup_chart function.
    view_df, cols, rows, color_lookup = setup_chart(subject, grouped=grouped, drop=drop)

    # Combine the original DataFrame with a new DataFrame that contains the data for all groups combined.
    combined = add_all(view_df)
    cols.append('all')

    # Assign colors to different groups in the strip plot.
    if diverging == True:
        color_lookup = assign_diverging_colors(cols)
    else:
        color_lookup = assign_random_colors(cols)

    # Create the strip plot using Plotly's px.strip function.
    fig = px.strip(combined, x=subject, y='group', color='group', color_discrete_map=color_lookup)

    # Update the layout of the strip plot.
    fig.update_layout(template='plotly_white')

    return fig

def make_hist(subject, grouped=False, drop='medium'):
    """
    Generate a histogram based on the specified subject (microbe or trait).

    Parameters:
        subject (str): The subject for which the histogram needs to be created.
        grouped (bool, optional): If True, the data will be grouped. Defaults to False.
        drop (str, optional): A category to be dropped from the 'cols'. Defaults to 'medium'.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object representing the histogram.
    """
    # Set up data and configurations for the chart using the setup_chart function.
    view_df, cols, rows, color_lookup = setup_chart(subject, grouped=grouped, drop=drop)

    # Combine the original DataFrame with a new DataFrame that contains the data for all groups combined.
    combined = add_all(view_df)
    cols.append('all')

    # Prepare data for the histogram.
    hist_data = []
    colors = [color_lookup[c] for c in cols]

    # Add data for each group to the hist_data list.
    for col in cols:
        hist_data.append(list(combined[combined['group'] == col][subject]))

    # Create the histogram using Plotly's ff.create_distplot function.
    fig = ff.create_distplot(hist_data, cols, bin_size=0.1, colors=colors, show_hist=True)

    # Update the layout of the histogram.
    fig.update_layout(template='plotly_white', title=subject)
    fig.update_yaxes(title='Number of Mice')
    fig.update_xaxes(title=subject)

    return fig

def make_abundance_plot(trait):
    """
    Generate a strip plot showing the abundance of different microbes (microbes) for a given trait.

    Parameters:
        trait (str): The trait for which the abundance plot needs to be created.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object representing the abundance plot.
    """
    # Get the leaf DataFrame for the given trait.
    leaf_df = get_leaf_df(trait)

    # Extract the names of microbes from the columns of the leaf DataFrame.
    gen_names = []
    for col in leaf_df.columns:
        if 'g__' in col:
            gen_names.append(col)

    # Create a new DataFrame with the abundance data for each microbe unstacked.
    leaf_df = pd.DataFrame(leaf_df.filter(gen_names).unstack())

    # Create the strip plot using Plotly's px.strip function.
    fig = px.strip(x=leaf_df.droplevel(1).index, y=list(leaf_df[0]))

    # Update the layout of the strip plot.
    fig.update_layout(template='plotly_white', title=trait[:-15].replace('_', ' '))
    fig.update_xaxes(title='Microbe')
    fig.update_yaxes(title='Abundance')

    return fig


import dash_bio as dashbio

def dendrogram(df, col, ascending=False):
    """
    Generate a clustergram (dendrogram) to visualize the comparison of microbe pairs based on a specified column.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data for microbe pairs.
        col (str): The column in the DataFrame to be used for the comparison.
        ascending (bool, optional): If True, the data will be sorted in ascending order. Defaults to False.

    Returns:
        dash_bio.Clustergram: A Dash Bio Clustergram object representing the dendrogram.
    """
    # Filter and sort the DataFrame based on the specified column.
    filtered = df.filter(items=['source', 'target', col]).sort_values(by=col, ascending=ascending)

    # Create a pivot table (matrix) from the filtered DataFrame.
    matrix = filtered.pivot('source', 'target', col)

    # Fill any missing values in the matrix with a value greater than the maximum (if ascending) or less than the minimum (if not ascending).
    filled = matrix.fillna((max(df[col]) + 1) if ascending else (min(df[col]) - 1))

    # Get the list of columns and rows from the filled matrix.
    columns = list(filled.columns.values)
    rows = list(filled.index)

    # Create the clustergram (dendrogram) using Dash Bio's Clustergram component.
    fig = dashbio.Clustergram(
        data=abs(filled.loc[rows].values),
        row_labels=rows,
        column_labels=columns,
        color_map='blues_r' if ascending else 'blues',
        optimal_leaf_order=True,
    )

    # Update the layout of the clustergram.
    fig.update_layout(
        width=800,
        height=800,
        title=f"<b>Comparison of Microbe Pairs<br>by {col.replace('_', ' ').title()}</b>"
    )

    return fig

def parse_trees(tree_paths):
    """
    Process the content of multiple tree files and return a summarized DataFrame with group information for each tree,
    along with a summary DataFrame that contains the verification information for each tree.

    Parameters:
        tree_paths (list): A list of file paths containing the tree data.

    Returns:
        pd.DataFrame: A DataFrame containing group information for all trees.
    """
    group_dfs = []  # A list to store DataFrames containing group information for each tree
    summary_df = pd.DataFrame(columns=['trait', 'inner', 'inner_checksum', 'outer', 'outer_checksum'])

    # Process each tree file
    for tree_path in tree_paths:
        trait = tree_path.split('/')[1]
        summary_row = {
            'trait': trait,
            'inner': None,
            'inner_checksum': False,
            'outer': None,
            'outer_checksum': False
        }

        group_df = pd.DataFrame(columns=['microbe', 'split', 'split_value', 'depth', 'isLeaf', 'y_mean', 'group_n', 'err', 'nobs'])

        # Read the content of the tree file and split it into lines
        with open(tree_path) as f:
            content = f.read().replace('"', '').split('\n')

        # Find the location of the root group in the content
        root_loc = None
        for ix, c in enumerate(content):
            if 'root' in c:
                root_loc = ix
                break

        # Process each line of the tree content
        leaf_count = 0
        inner_count = 0
        for line in content[root_loc:-4]:
            simplified = line[6:].strip().replace(',', '').replace('(', '').replace(')', '')
            words = simplified.split(' ')
            row = {
                'microbe': None,
                'split': None,
                'split_value': None,
                'depth': None,
                'isLeaf': False,
                'y_mean': None,
                'group_n': 1,
                'nobs': None,
                'err': None,
            }
            row['depth'] = simplified.count('|')

            for ix, w in enumerate(words):
                if w in ['<=', '<', '>']:
                    row['split'] = w
                    row['split_value'] = float(words[ix + 1].replace(':', ''))
                if "g__" in w:
                    row['microbe'] = w
                if w == 'n':
                    row['isLeaf'] = True
                    row['y_mean'] = float(words[ix - 1])
                    row['nobs'] = int(words[ix + 2])
                    row['err'] = float(words[ix + 5])
                if len(w) > 0:
                    if (w[0] == '[') and (w[-1] == ']'):
                        row['group_n'] = int(w.replace(']', '').replace('[', ''))

            if row['isLeaf'] == False:
                inner_count += 1
            else:
                leaf_count += 1

            # Concatenate the row to the group DataFrame
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                group_df = pd.concat([group_df, pd.DataFrame(row, index=[0])], ignore_index=True)

        # Extract the verification information for inner and outer groups
        verification_strings = content[-3:-1]
        for s in verification_strings:
            words = s.split(' ')
            stat = words[3]
            val = words[-1]
            checksum = None
            if stat == 'inner':
                summary_row['inner'] = val
                summary_row['inner_checksum'] = ((int(val) - inner_count) == 0)
            if stat == 'terminal':
                summary_row['outer'] = val
                summary_row['outer_checksum'] = ((int(val) - leaf_count) == 0)

        # Add the trait column to the group DataFrame and append it to the list
        group_df['trait'] = trait
        group_dfs.append(group_df)

        # Concatenate the summary_row to the summary DataFrame
        summary_df = pd.concat([summary_df, pd.Series(summary_row).to_frame().T], ignore_index=True)

    # Concatenate all the group DataFrames into a single DataFrame
    all_groups = pd.concat(group_dfs, ignore_index=True)
    return all_groups


def find_parent(group_n, depth, single_tree):
    """
    Find the parent group of a given group within a tree.

    Parameters:
        group_n (int): Group number of the group for which to find the parent.
        depth (int): Depth of the group within the tree.
        single_tree (pd.DataFrame): DataFrame containing group information for a single tree.

    Returns:
        int: Group number of the parent group, or 1 if the group is at depth 0 (root).
    """
    if depth > 0:
        # Create a copy of the DataFrame containing groups one level above the given depth
        one_aboves = single_tree[single_tree['depth'] == depth - 1].copy()
        group_list = one_aboves['group_n']
        # Filter the list of groups to include only those with group numbers less than the given group_n
        filt_list = [n for n in group_list if n < group_n]
        # Return the group number of the last (highest) group in the filtered list,
        # which corresponds to the parent group of the given group
        return filt_list[-1]
    else:
        # If the group is at depth 0 (root), return group number 1, which indicates the root group itself.
        return 1

def get_tree(trait, all_groups):
    """
    Get information about a single tree for a specific trait.

    Parameters:
        trait (str): The name of the trait.
        all_groups (DataFrame): DataFrame containing information about all trees.

    Returns:
        DataFrame: DataFrame containing information about the single tree for the given trait.
    """
    # Filter the DataFrame to get the groups for the specified trait
    single_tree = all_groups[all_groups['trait'] == trait]

    # Calculate the number of observations (nobs) for each group and handle missing values
    nobs = list(single_tree['nobs'])
    nobs.reverse()
    for i, nob in enumerate(nobs):
        if nob is None:
            nobs[i] = nobs[i-1] + nobs[i-2]
    nobs.reverse()
    single_tree['nobs'] = nobs

    # Calculate the parent group for each group in the tree
    single_tree['parent'] = [find_parent(n, d, single_tree) for n, d in zip(single_tree['group_n'], single_tree['depth'])]

    # Remove groups with missing microbe information
    single_tree = single_tree[single_tree['microbe'].notna()]

    # Create a color mapping based on the y_mean values
    colormap = cm.get_cmap('RdYlGn', 100)
    m = interpolate.interp1d([-2, 2], [0, 1])
    single_tree['color'] = [f'rgb({colormap(m(a))[0]},{colormap(m(a))[1]},{colormap(m(a))[2]})' for a in single_tree['y_mean']]

    return single_tree


def process_trees(traits):
    """
    Process the trees for the given traits, create tree plots, and save the results.

    Parameters:
        traits (list): A list of trait names.

    Returns:
        DataFrame: A DataFrame containing the processed tree data for all traits.
    """
    processed_trees = []

    # Loop through each trait and create tree plots
    for trait in traits:
        processed_trees.append(get_tree(trait))

    # Combine the processed tree data into a single DataFrame
    processed_df = pd.concat(processed_trees)

    # Save the DataFrame to a CSV and a pickle file
    processed_df.to_csv('new_processed_tree.csv', index=False)
    processed_df.to_pickle('new_processed_tree.pkl')

    return processed_df


def get_microbe_co(processed_df):
    """
    Get co-occurrence information for microbes and traits from the processed DataFrame.

    Parameters:
        processed_df (DataFrame): A DataFrame containing processed tree data for traits.

    Returns:
        DataFrame: A DataFrame containing co-occurrence information for microbes and traits.
    """
    # Group traits by microbe and create a list of unique traits for each microbe
    co_occurrence = processed_df.groupby('microbe')['trait'].apply(list)
    co_occurrence = co_occurrence.reset_index(name='trait')

    # Remove duplicates from the list of traits for each microbe
    co_occurrence['trait'] = [list(set(g)) for g in co_occurrence['trait']]

    # Calculate the count of unique traits for each microbe
    co_occurrence['count'] = [len(g) for g in co_occurrence['trait']]

    # Sort the DataFrame by the count of unique traits in descending order
    co_occurrence = co_occurrence.sort_values(by='count', ascending=False)

    return co_occurrence


def get_trait_co_lookup(processed_df):
    """
    Get co-occurrence lookup for traits and associated microbes from the processed DataFrame.

    Parameters:
        processed_df (DataFrame): A DataFrame containing processed tree data for traits and microbes.

    Returns:
        dict: A dictionary containing the co-occurrence lookup for traits and their associated microbes.
              The dictionary has traits as keys and lists of associated microbes as values.
    """
    # Group microbes by trait and create a list of associated microbes for each trait
    b_co = processed_df.groupby('trait')['microbe'].apply(list).reset_index(name='microbe').set_index('trait')

    # Convert the DataFrame to a dictionary for easy lookup
    b_co_lookup = b_co['microbe'].to_dict()

    return b_co_lookup


def find_co(source, target, b_co_lookup):
    """
    Find traits where both the given source and target microbes co-occur.

    Parameters:
        source (str): The name of the source microbe.
        target (str): The name of the target microbe.
        b_co_lookup (dict): A dictionary containing the co-occurrence lookup for traits and their associated microbes.

    Returns:
        list: A list of traits where both the source and target microbes co-occur.
    """
    co_list = []
    for trait in b_co_lookup.keys():
        g = b_co_lookup[trait]
        if (source in g) and (target in g):
            co_list.append(trait)
    return co_list


def get_stack(processed_df):
    """
    Generate a DataFrame representing the co-occurrence count of microbes across traits.

    Parameters:
        processed_df (DataFrame): A DataFrame containing processed tree data for traits and microbes.

    Returns:
        DataFrame: A DataFrame containing the co-occurrence count of microbes across traits.
                   The DataFrame has columns 'source', 'target', 'count', and 'traits'.
                   'source' and 'target' represent the microbes that co-occur across traits.
                   'count' represents the number of traits in which the microbes co-occur.
                   'traits' contains the list of traits where the microbes co-occur.
    """
    microbes_list = list(processed_df.groupby('trait')['microbe'].apply(list).reset_index(name='microbe')['microbe'])
    microbes_list = [list(set(l)) for l in microbes_list]
    u = (pd.get_dummies(pd.DataFrame(microbes_list), prefix='', prefix_sep='')
       .groupby(level=0, axis=1)
       .sum())

    v = u.T.dot(u)
    v.values[(np.r_[:len(v)], ) * 2] = 0

    stack = pd.DataFrame(pd.DataFrame(v).stack()).reset_index()
    stack = stack.rename(columns={'level_0':'source', 'level_1':'target', 0:'count'}).sort_values(by='count', ascending=False)
    stack = stack[stack['count']>0]

    # Find traits where each source and target microbes co-occur
    stack['traits'] = [find_co(source, target) for source, target in zip(stack['source'], stack['target'])]

    # Save the DataFrame to CSV and pickle files
    stack.to_csv('edgelist.csv', index=False)
    stack.to_pickle('edgelist.pkl')

    return stack


def get_lookup(single_tree):
    """
    Generate a lookup dictionary for groups in a single decision tree.

    Parameters:
        single_tree (DataFrame): A DataFrame containing data for a single decision tree.

    Returns:
        dict: A lookup dictionary where the keys are group numbers (group_n) and the values are dictionaries
              containing information about each group.
              Each group dictionary contains the following keys:
                - 'color': The color associated with the group in visualizations.
                - 'nobs': The number of observations (mice) in the group.
                - 'y_mean': The mean value of the trait ranknorm associated with the group.
                - 'err': The error value associated with the group.
                - 'splitting_on': The microbe that the group is splitting on. For leaf groups, it is 'LEAF'.
                - 'label_short': A short label for the group, suitable for visualization.
                - 'label_long': A longer label for the group, suitable for tooltips in visualizations.
    """
    lookup = single_tree.set_index('group_n').to_dict(orient='index')

    # Add information for root group and any group without children (leaves)
    for n in [0, 1]:
        lookup[n] = {
            'color': 'black',
            'nobs': 375,  # Update with the actual number of observations
            'y_mean': None,  # Update with the actual y_mean value
            'err': None  # Update with the actual error value
        }

    for group_n in lookup.keys():
        children = single_tree[single_tree['parent'] == group_n]
        if children.shape[0] > 0:
            splitting_on = list(children['microbe'])[0][3:]
        else:
            splitting_on = 'LEAF'

        lookup[group_n]['splitting_on'] = splitting_on

        if splitting_on == 'LEAF':
            lookup[group_n]['label_short'] = f"Cohort {group_n}"
            lookup[group_n]['label_long'] = f'Group {group_n} has {lookup[group_n]["nobs"]} mice<br />with a mean {list(single_tree["trait"])[0].replace("_ranknorm","")} of<br />{lookup[group_n]["y_mean"]}<extra></extra>'
        else:
            lookup[group_n]['label_short'] = splitting_on.replace('g__', '').replace('_', ' ')
            lookup[group_n]['label_long'] = f'Group {group_n} has {lookup[group_n]["nobs"]} mice<br />and is splitting on {splitting_on}<extra></extra>'

    return lookup



def make_network(selected):
    """
    Create a network visualization for tree co-occurrence.

    Parameters:
        selected (str): The selected microbe to visualize tree co-occurrence.

    Returns:
        plotly.graph_objs._figure.Figure: A Plotly figure object representing the network visualization.
    """
    # Load the pre-processed edgelist data
    stack = pd.read_pickle('edgelist.pkl').rename(columns={'count': 'weight'})

    # Calculate the sizes of the groups based on their weights
    sizes = stack.groupby('source').sum().sort_values(by='weight', ascending=False)['weight'].to_dict()

    # Create a graph using networkx from the edgelist data
    G = nx.from_pandas_edgelist(stack, "source", "target", ["weight", "traits"])

    # Perform spring layout algorithm on the graph to get positions of groups
    pos = nx.spring_layout(G, weight='weight', seed=7)

    # Initialize a list to store edge traces
    edge_traces = []

    # Filter the edgelist data to include only the edges connected to the selected microbe
    subset = stack[stack['source'] == selected]

    # Iterate over all edges in the graph
    for edge in G.edges.data():
        # Get the weight of the edge
        weight = edge[2]['weight']
        # Set color for the edge based on whether the selected microbe is connected to it
        if selected in [edge[0], edge[1]]:
            color = "red"  # Selected microbe connected
        else:
            color = f'rgba(0,0,0,{weight*0.01})'  # Non-selected microbe connected with transparency based on weight

        # Create a trace for the edge
        trace = go.Scatter(
            x=[pos[edge[0]][0], pos[edge[1]][0]],
            y=[pos[edge[0]][1], pos[edge[1]][1]],
            mode="lines",
            hoverinfo=None,
            line=dict(width=weight * 2, color=color)  # Set line width and color
        )
        edge_traces.append(trace)

    # Initialize lists to store group positions, names, sizes, and colors
    group_x = []
    group_y = []
    group_name = []
    group_sizes = []
    group_colors = []

    # Iterate over all groups in the graph
    for group in G.groups():
        group_x.append(pos[group][0])  # X position
        group_y.append(pos[group][1])  # Y position
        group_name.append(group)  # Group name
        group_sizes.append(np.sqrt(sizes[group]))  # Size of the group
        # Set group color based on whether the group is connected to the selected microbe or not
        group_colors.append('red' if group in list(subset['target']) else 'black' if group == selected else '#ddd')

    # Create a scatter trace for groups
    group_trace = go.Scatter(
        x=group_x, y=group_y,
        mode='markers',
        text=group_name,
        hoverinfo='text',
        marker=dict(size=group_sizes, color=group_colors)
    )

    # Create the figure with edge traces and group trace
    fig = go.Figure(data=edge_traces + [group_trace],
                    layout=go.Layout(
                        title=f'Tree Co-Occurrence for {selected if selected else "all microbes"}',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))

    fig.update_layout(template='plotly_white', height=500, width=500, clickmode='event+select')
    return fig

def make_related_bar(selected):
    """
    Create a bar plot representing microbes related to the selected microbe by the number of trees they co-occur in.

    Parameters:
        selected (str): The selected microbe for which related microbes are visualized.

    Returns:
        plotly.graph_objs._figure.Figure: A Plotly figure object representing the bar plot.
    """
    # Load the pre-processed edgelist data
    stack = pd.read_pickle('edgelist.pkl').rename(columns={'count': 'weight'})

    # Filter the edgelist data to include only the edges connected to the selected microbe
    subset = stack[stack['source'] == selected].sort_values(by='weight')

    # Create a bar plot using Plotly
    fig = px.bar(subset, x='target', y='weight')

    # Set the hover template to display the number of trees the microbes co-occur with the selected microbe
    fig.update_traces(hovertemplate='%{x} co-occurs with<br>' + selected + ' in %{y} trees')

    fig.update_layout(template="plotly_white",
                      title=f'Microbes related to {selected}<br>by Number of Trees',
                      clickmode='event+select'
                      )
    fig.update_xaxes(title='Related gene')
    fig.update_yaxes(title='Number of Trees')
    return fig

def make_sankey(trait, tree):
    """
    Create a Sankey diagram representing the conditional inference tree for the specified trait.

    Parameters:
        trait (str): The trait for which the conditional inference tree is visualized.
        tree (pd.DataFrame): The processed DataFrame containing information about the tree groups.

    Returns:
        plotly.graph_objs._figure.Figure: A Plotly figure object representing the Sankey diagram.
    """
    # Get the single tree for the specified trait
    single_tree = get_tree(trait, tree)

    # Get the lookup dictionary for group information
    lookup = get_lookup(single_tree)

    # Create a list of labels for groups in the Sankey diagram
    labels = list(range(0, max(single_tree['group_n']) + 1))

    # Create a Sankey diagram using Plotly
    fig = go.Figure(data=[go.Sankey(
        # arrangement='snap',
        group=dict(
            pad=15,
            thickness=20,
            line=dict(color=[lookup[l]['color'] for l in labels], width=0.5),
            label=[lookup[l]['label_short'] for l in labels],
            customdata=[lookup[l]['label_long'] for l in labels],
            hovertemplate='%{customdata}',  # Set custom hover template for groups
            color=[lookup[l]['color'] for l in labels],
        ),
        link=dict(
            arrowlen=15,
            source=single_tree['parent'],  # Indices correspond to labels, e.g., A1, A2, A1, B1, ...
            target=single_tree['group_n'],
            value=single_tree['nobs'],
            customdata=[f'{g[3:]} {s} {sv}' for s, sv, g in
                        zip(single_tree['split'], single_tree['split_value'], single_tree['microbe'])],
            hovertemplate='There are %{value} mice in this branch<br />with %{customdata}'
        ))])

    fig.update_layout(title_text=f"Conditional Inference Tree for {trait.replace('_ranknorm', '')}", font_size=10)
    return fig



def make_pcoord(reference, tree):
    """
    Generate a parallel coordinates plot based on the reference (trait or microbe) data.

    Parameters:
        reference (str): The reference (trait or microbe) for which the parallel coordinates plot is generated.
        tree (pd.DataFrame): The processed DataFrame containing information about the tree groups.

    Returns:
        plotly.graph_objs._figure.Figure: A Plotly figure object representing the parallel coordinates plot.
    """
    # Load trait and abundance data
    trait_data = read_trait()[0]
    abundance_data = read_abundance()[0]

    # Check if the reference is a microbe or trait
    if 'g__' in reference:
        # If it is a microbe, create a DataFrame with abundance data for the given microbe
        df = get_abundance_df(reference)
        # Set the color for the parallel coordinates plot to the second column of the DataFrame
        color = df.columns[1]
        # dims = list(df.columns[4:])
        # dims.append(df.columns[1])
        # labels = [{d:d.replace('_batch_ranknorm','')} for d in dims]
        # print(labels)
    else:
        # If it is a trait, create a DataFrame with trait data for the given trait
        df = get_trait_df(reference, abundance_data=abundance_data)
        # Get the tree for the given trait
        single_tree = get_tree(reference, tree)
        # Determine the order of microbes for the Sankey diagram based on the tree
        sankey_order = []
        for g in list(single_tree['microbe']):
            if g not in sankey_order:
                sankey_order.append(g)
        # print(sankey_order)
        # print(list(df.columns.values))
        # Filter the DataFrame to include only the microbes specified in the Sankey diagram order
        df = df[sankey_order]
        # Add the trait column to the DataFrame
        df[reference] = list(trait_data.filter(items=[reference])[reference])
        # Reset the index of the DataFrame and drop the old index column
        df = df.reset_index().drop(columns=['index'])
        # Convert all columns to numeric values, ignoring non-numeric entries
        df = df.apply(pd.to_numeric, errors='coerce')
        # Set the color for the parallel coordinates plot to the reference trait
        color = reference
        # Get the list of dimensions (microbes and trait) for labeling purposes
        dims = list(df.columns)
        # Create a dictionary to map dimensions to their corresponding labels for the plot
        labels = {}
        for d in dims:
            labels[d] = d.replace('g__', '').replace(reference, 'trait')
        # print(labels)

    # Create the parallel coordinates plot using Plotly
    fig = px.parallel_coordinates(df, color=color, labels=labels,
                                  color_continuous_scale=px.colors.diverging.RdYlGn,
                                  color_continuous_midpoint=0)

    # Rotate the labels on the parallel coordinates plot for better visibility
    fig.data[0]['labelangle'] = -45

    # Update the layout of the plot
    fig.update_layout(margin={'t': 400}, height=800, title=clean(reference))

    return fig