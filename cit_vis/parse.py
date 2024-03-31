import os
import dotenv
import pandas as pd
import json

def get_trait_folders(data_dir, debug=True):
    """Get the folders in the data directory, which correspond to the traits.
    
    Args:
        data_dir (str): The path to the data directory.
        debug (bool): Whether to print debug statements.
        
    Returns:
        list: A list of the folders in the data directory."""
    return [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

def aggregate_by_group(mice, trait, data_dir, debug=True):
    """Aggregate the mice data by group.

    Args:
        mice (pd.DataFrame): The mice data.
        trait (str): The trait to aggregate by.
        data_dir (str): The path to the data directory.
        debug (bool): Whether to print debug statements.
    
    Returns:
        pd.DataFrame: The aggregated data by group.
    """
    if debug: print(f"Aggregating by group for {trait}...")

    # Aggregate by group
    groups = mice.groupby("Group ID").agg({
        "Mouse ID": lambda x: list(x),
        "Feature Path": "first",
        trait: "mean"})
    groups.reset_index(inplace=True)
    groups.columns = ["Group ID", "Mouse IDs", "Feature Path", "Mean Trait Value"]

    # Split the feature path
    groups["Feature Path"] = [path.split(" -> ") for path in groups["Feature Path"]]
    groups.sort_values("Mean Trait Value", ascending=False, inplace=True)
    groups.set_index("Group ID", inplace=True)

    # Save the groups
    groups.to_pickle(os.path.join(data_dir, trait, "groups.pkl"))
    if debug: print(f"Groups saved to {os.path.join(data_dir, trait, 'groups.pkl')}.")
    return groups

def get_grid_structure(groups, rules, debug=True):
    """Create a grid structure for the data.

    Args:
        groups (pd.DataFrame): The aggregated data by group.
        rules (pd.DataFrame): The rules for the data.
        debug (bool): Whether to print debug statements.

    Returns:
        pd.DataFrame: The grid structure for the data.
    """
    if debug: print("Creating grid structure...")

    # Create the grid structure
    groups_dict = groups.copy().to_dict(orient="index")
    cols = []
    group_names = groups_dict.keys()
    feature_names = []

    # Get the unique feature names (microbes)
    for path in groups["Feature Path"]:
        for feature in path:
            if feature not in feature_names:
                feature_names.append(feature)

    # Create the grid
    for group, data in groups_dict.items():
        end_branch = False
        row = []

        # Create the row for the group
        for feature in feature_names:
            split = rules[rules["Microbe"]==feature]["Split"].values[0]
            value = rules[rules["Microbe"]==feature]["Value"].values[0]
            if feature in data["Feature Path"]:
                row.append({
                    "Type": "Plot",
                    "Feature": feature,
                    "Split": split,
                    "Value": value,
                    "Mouse IDs": data["Mouse IDs"],
                })
                if feature == data["Feature Path"][-1]:
                    end_branch = True
            else:
                if end_branch==True:
                    row.append({
                        "Type": "Empty",
                        "Feature": feature,
                         "Split": split,
                         "Value": value,
                        "Mouse IDs": [],
                    })
                else:
                    row.append({
                        "Type": "Arrow",
                        "Feature": feature,
                        "Split": split,
                        "Value": value,
                        "Mouse IDs": [],
                    })
        cols.append(row)

    grid = pd.DataFrame(cols, columns=feature_names, index=group_names).T
    if debug: print("Grid structure created.")
    return grid
        
def get_edgelist(trait, data_dir, debug=True):
    """Create an edgelist for the trait.

    Args:
        trait (str): The trait to create an edgelist for.
        data_dir (str): The path to the data directory.
        debug (bool): Whether to print debug statements.

    Returns:
        pd.DataFrame: The edgelist for the trait.
    """
    if debug: print(f"Creating edgelist for {trait}...")
    rules = pd.read_csv(os.path.join(data_dir, trait, "rules.csv"))
    mice = pd.read_csv(os.path.join(data_dir, trait, "mice.csv"))

    # split the feature path
    mice["Feature Path"] = mice["Feature Path"].apply(lambda x: x.split(" -> "))

    # extend the paths to include the group id
    extended_paths = []
    for i, path in enumerate(mice["Feature Path"]):
        path.append(mice["Group ID"][i])
        if path not in extended_paths:
            extended_paths.append(path)

    # iterate over all the paths to get the edges
    edges = []
    for path in extended_paths:
        for i in range(len(path)-1):
            group = path[-1]
            if path[i+1] == group:
                # Is a leaf node
                mean_trait = mice[mice["Group ID"]==group][trait].mean()
            else:
                # Is not a leaf node
                mean_trait = None
            edge = {
                "source": path[i],
                "target": path[i+1],
                "split": rules[rules["Microbe"]==path[i]]["Split"].values[0],
                "group": group,
                "value": mice[mice["Group ID"]==group].shape[0],
                "mean_trait": mean_trait,
            }
            if edge["source"] != edge["target"]:
                edges.append(edge)

    # create the edgelist
    edgelist = pd.DataFrame(edges)
    edgelist.sort_values("mean_trait", ascending=False, inplace=True)

    # save the edgelist
    edgelist.to_csv(os.path.join(data_dir, trait, "edgelist.csv"), index=False)
    if debug: print(f"Edgelist saved to {os.path.join(data_dir, trait, 'edgelist.csv')}.")
    return edgelist

def find_relevant(microbe, data_dir, debug=True):
    """Find the relevant traits for a microbe.

    Args:
        microbe (str): The microbe to find relevant traits for.
        data_dir (str): The path to the data directory.
        debug (bool): Whether to print debug statements.
    
    Returns:
        list: The relevant traits for the microbe.
    """
    if debug: print(f"Finding relevant traits for {microbe}...")

    # Load the relevance json
    with open(os.path.join(data_dir, "relevance.json"), "r") as f:
        relevant = json.load(f)[microbe]
    if debug: print(f"Relevant traits for {microbe}: {relevant}")
    return relevant

def parse_data(debug=True):
    """Parse the data in the data directory.

    Args:
        debug (bool): Whether to print debug statements.

    Returns:
        None
    """
    if debug: print("Parsing data...")

    # Load the data directory
    dotenv.load_dotenv()
    data_dir = os.getenv("DATA_DIR")

    # Get the folders in the data directory
    folders = get_trait_folders(data_dir, debug=debug)
    relevance = {}

    # Parse the data for each trait
    for trait in folders:
        if debug: print(f"Parsing data for {trait}...")
        mice = pd.read_csv(os.path.join(data_dir, trait, "mice.csv"))
        groups = aggregate_by_group(mice, trait, data_dir, debug=debug)
        rules = pd.read_csv(os.path.join(data_dir, trait, "rules.csv"))

        # Add the relevant traits for each microbe
        for microbe in rules["Microbe"]:
            if microbe not in relevance.keys():
                relevance[microbe] = [trait]
            else:
                relevance[microbe].append(trait)
        
        # Create the grid for each trait
        grid = get_grid_structure(groups, rules, debug=debug)
        grid.to_pickle(os.path.join(data_dir, trait, "grid.pkl"))

    # Save the relevance summary in the main data directory
    with open(os.path.join(data_dir, "relevance.json"), "w") as f:
        json.dump(relevance, f)
    if debug: print(f"Data parsed. Saved relevance to {os.path.join(data_dir, 'relevance.json')}.")

if __name__ == "__main__":
    parse_data()