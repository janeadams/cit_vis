import os
import dotenv
import pandas as pd

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

def parse_rules(trait, data_dir):
    tree_rules = open(os.path.join(data_dir, trait, "rules.txt")).read()
    rules_df = pd.DataFrame(columns=['Depth', 'Microbe', 'Split', 'Value'])
    last_at_depth = {0: 'Root'}
    for i, line in enumerate(tree_rules.split("\n")):
        depth = line.count('|')
        elements = line.split( )[depth:]
        if len(elements) == 3:
            if elements[0] != 'class:':
                microbe, split, value = elements
                rules_df.loc[i] = [depth, microbe, split, value]
        last_at_depth[depth] = microbe
    os.makedirs(os.path.join(data_dir, trait), exist_ok=True)
    rules_df.to_csv(os.path.join(data_dir, trait, "rules.csv"), index=False)
    return rules_df

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
        
def get_edgelist(groups):
    edgelist = []
    for i, group in groups.iterrows():
        for i in range(len(group["Feature Path"])-1):
            source = group["Feature Path"][i]
            target = group["Feature Path"][i+1]
            value = len(group.index)
            mean_trait = group["Mean Trait Value"]
            edgelist.append([source, target, value, mean_trait])
    edgeDF = pd.DataFrame(edgelist, columns=["source", "target", "value", "mean_trait"])
    return edgeDF

def parse_data():
    dotenv.load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    folders = get_trait_folders(data_dir)
    for trait in folders:
        mice = pd.read_csv(os.path.join(data_dir, trait, "mice.csv"))
        groups = aggregate_by_group(mice, trait, data_dir)
        rules = parse_rules(trait, data_dir)
        grid = get_grid_structure(groups, rules)
        grid.to_pickle(os.path.join(data_dir, trait, "grid.pkl"))

if __name__ == "__main__":
    parse_data()