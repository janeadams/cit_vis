import pandas as pd
import json

def get_assay_scores(assay=None):
    """Load the assay scores from file
    Args:
        assay (str): The name of the assay
    Returns:
        pd.Series: The assay scores
    """
    if assay:
        return pd.read_csv(f'data/{assay}/mice.csv', index_col=0)
    return pd.read_csv('data/assay_scores.csv', index_col=0)

def group_by_nodes(df, assay=None):
    if assay:
        # See if it's cached:
        try:
            return pd.read_csv(f"data/{assay}/nodes.csv", index_col=0)
        except FileNotFoundError:
            pass
    node_summary = df.groupby('Node_ID').agg({
        'Assay': ['mean', 'count'],
        'Path': 'first'
    })
    node_summary.columns = ['Mean', 'Count', 'Path']
    if assay: node_summary.to_csv(f"data/{assay}/nodes.csv")
    return node_summary

def create_matrix(df, assay):
    node_summary = group_by_nodes(df)
    node_genus_matrix = node_summary.explode('Path').reset_index().pivot_table(index='Node_ID', columns='Path', values='Count', fill_value=0)
    node_genus_matrix.columns = [col.split('>')[-2] for col in node_genus_matrix.columns]
    node_genus_matrix.to_csv(f"data/{assay}/matrix.csv")
    return node_genus_matrix

def get_relevant_genera(assay):
    """Load the relevant genera from file
    Args:
        assay (str): The name of the assay
    Returns:
        list: The relevant genera
    """
    with open("data/relevant_genera.json", "r") as json_file:
        relevant_genera = json.load(json_file)
    return list(relevant_genera[assay])

def get_relevant_assays(genus):
    """Load the relevant assays from file
    Args:
        genus (str): The name of the genus
    Returns:
        list: The relevant assays
    """
    with open("data/relevant_assays.json", "r") as json_file:
        relevant_assays = json.load(json_file)
    return list(relevant_assays[genus])

def load_tree(assay):
    """Load the decision tree from file
    Args:
        assay (str): The name of the assay
    Returns:
        str: The decision tree rules
    """
    with open(f"data/{assay}/rules.txt", "r") as text_file:
        tree_rules = text_file.read()
    return tree_rules

def get_mice(assay, node_id):
    """Get the mouse IDs for a given node
    Args:
        assay (str): The name of the assay
        node_id (int): The node ID
    Returns:
        pd.Series: The mouse IDs
    """
    mice_df = pd.read_csv(f'data/{assay}/mice.csv')
    cohort = mice_df[mice_df['Node_ID'] == node_id]['Mouse_ID']
    return cohort

def get_genus_abundances(assay=None, node_id=None, genus=None):
    """Get the genus abundances for a given assay
    Args:
        assay (str): The name of the assay
        node_id (int): The node ID
        genus (str): The name of the genus
    Returns:
        pd.Series: The genus abundances
    """
    genus_abundances_df = pd.read_csv('data/genus_abundances.csv', index_col=0)
    mice = None
    if assay:
        relevant_genera = get_relevant_genera(assay)
        if node_id:
            # Get a list of mice for this node
            mice = get_mice(assay, node_id)
            # Filter by those mice
            genus_abundances_df = genus_abundances_df.loc[mice]
        # Filter by relevant genera
        return genus_abundances_df[relevant_genera]
    if genus:
        # Get the subset for this genus
        subset = genus_abundances_df[genus]
        if mice:
            return subset.loc[mice]
        return subset
    return genus_abundances_df

def get_split_conditions(assay, genus):
    """Get the split value for a given node and genus
    Args:
        assay (str): The name of the assay
        genus (str): The name of the genus
    Returns:
        float: The split value
        str: The split condition
    """
    tree_rules = load_tree(assay)
    split_values = []
    split_conditions = []
    for line in tree_rules.split("\n"):
        if genus in line:
            split_conditions.append(line.split()[-2])
            split_values.append(float(line.split()[-1]))
    return split_values, split_conditions

def is_relevant(assay, node_ID, genus):
    """Check if a genus is relevant for a given node
    Args:
        assay (str): The name of the assay
        node_ID (int): The node ID
        genus (str): The name of the genus
    Returns:
        bool: Whether the genus is relevant
    """
    matrix = pd.read_csv(f"data/{assay}/matrix.csv", index_col=0)
    return genus in matrix.columns and matrix.loc[genus, node_ID] > 0