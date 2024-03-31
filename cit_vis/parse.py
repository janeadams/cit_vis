import os
import dotenv
import pandas as pd
import json

def get_trait_folders(data_dir):
    return [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

def aggregate_by_group(mice, trait, data_dir):
    groups = mice.groupby("Group ID").agg({
        "Mouse ID": lambda x: list(x),
        "Feature Path": "first",
        trait: "mean"})
    groups.reset_index(inplace=True)
    groups.columns = ["Group ID", "Mouse IDs", "Feature Path", "Mean Trait Value"]
    groups["Feature Path"] = [path.split(" -> ") for path in groups["Feature Path"]]
    groups.sort_values("Mean Trait Value", ascending=False, inplace=True)
    groups.set_index("Group ID", inplace=True)
    groups.to_pickle(os.path.join(data_dir, trait, "groups.pkl"))
    return groups

def get_grid_structure(groups, rules):
    groups_dict = groups.copy().to_dict(orient="index")
    cols = []
    group_names = groups_dict.keys()
    feature_names = []
    for path in groups["Feature Path"]:
        for feature in path:
            if feature not in feature_names:
                feature_names.append(feature)
    for group, data in groups_dict.items():
        end_branch = False
        row = []
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
    return grid
        
def get_edgelist(trait, data_dir):
    rules = pd.read_csv(os.path.join(data_dir, trait, "rules.csv"))
    mice = pd.read_csv(os.path.join(data_dir, trait, "mice.csv"))
    mice["Feature Path"] = mice["Feature Path"].apply(lambda x: x.split(" -> "))
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
    edgelist = pd.DataFrame(edges)
    edgelist.sort_values("mean_trait", ascending=False, inplace=True)
    edgelist.to_csv(os.path.join(data_dir, trait, "edgelist.csv"), index=False)
    return edgelist

def find_relevant(microbe, data_dir):
    with open(os.path.join(data_dir, "relevance.json"), "r") as f:
        relevance = json.load(f)
    return relevance[microbe]

def parse_data():
    dotenv.load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    folders = get_trait_folders(data_dir)
    relevance = {}
    for trait in folders:
        mice = pd.read_csv(os.path.join(data_dir, trait, "mice.csv"))
        groups = aggregate_by_group(mice, trait, data_dir)
        rules = pd.read_csv(os.path.join(data_dir, trait, "rules.csv"))
        for microbe in rules["Microbe"]:
            if microbe not in relevance.keys():
                relevance[microbe] = [trait]
            else:
                relevance[microbe].append(trait)
        grid = get_grid_structure(groups, rules)
        grid.to_pickle(os.path.join(data_dir, trait, "grid.pkl"))
    with open(os.path.join(data_dir, "relevance.json"), "w") as f:
        json.dump(relevance, f)

if __name__ == "__main__":
    parse_data()