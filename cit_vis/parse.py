import os
import dotenv
import pandas as pd

def get_trait_folders(data_dir):
    return [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

def parse_rules(trait, data_dir):
    tree_rules = open(os.path.join(data_dir, trait, "rules.txt")).read()
    rules_df = pd.DataFrame(columns=['Depth', 'Microbe', 'Split', 'Value', 'Parent', 'Children'])
    last_at_depth = {0: 'Root'}
    for i, line in enumerate(tree_rules.split("\n")):
        depth = line.count('|')
        elements = line.split( )[depth:]
        if len(elements) == 3:
            if elements[0] != 'class:':
                microbe, split, value = elements
                microbe = microbe+' '+split
                parent = last_at_depth[depth-1]
                rules_df.loc[i] = [depth, microbe, split, value, parent, []]
        last_at_depth[depth] = microbe
    for i, rule in rules_df.iterrows():
        # Find the indices of the parent:
        parent = rule['Parent']
        parent_indices = rules_df[rules_df['Microbe'] == parent]
        for parent_index in parent_indices.index:
            rules_df.at[parent_index, 'Children'].append(rule['Microbe'])
    os.makedirs(os.path.join(data_dir, trait), exist_ok=True)
    rules_df.to_csv(os.path.join(data_dir, trait, "_rules.csv"), index=False)
    return rules_df

def apply_rules(rules, microbes, data_dir, trait):
    # Convert rule rows to a list of dictionaries
    rules_dict = rules.set_index('Microbe').to_dict(orient='index')
    print(rules_dict)
    
    # Recurse through the tree to assign a list of mice to each rule:
    def recurse_tree(rule, mice):
        if len(rule['Children'])>0:
            for child in rule['Children']:
                child_rule = rules_dict[child]
                recurse_tree(child_rule, mice)
        else:
            mice_in_rule = microbes[(microbes[rule['Microbe']] <= float(rule['Value'])) if rule['Split'] == '<=' else (microbes[rule['Microbe']] > float(rule['Value']))].index
            mice_in_rule = [m for m in mice if m in mice_in_rule]
            rule['Mice'] = mice_in_rule

    recurse_tree(rules_dict[0], microbes.index)
    # Create a DataFrame with the mice in each rule and save it
    mice_in_rules_df = pd.DataFrame(columns=['Rule', 'Mice'])
    for rule in rules_dict.items():
        mice_in_rules_df.loc[len(mice_in_rules_df)] = [rule['Microbe'], rule['Mice']]
    mice_in_rules_df.to_csv(os.path.join(data_dir, trait, "mice_in_rules.csv"), index=False)
    return mice_in_rules_df
        
        

def parse_data():
    dotenv.load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    folders = get_trait_folders(data_dir)
    microbes = pd.read_csv(os.path.join(data_dir, "microbe_abundances.csv"), index_col=0)
    for trait in folders:
        rules = parse_rules(trait, data_dir)
        levels = apply_rules(rules, microbes, data_dir, trait)
    return levels

if __name__ == "__main__":
    parse_data()