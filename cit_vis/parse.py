import os
import dotenv
import pandas as pd

def get_trait_folders(data_dir):
    return [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

def aggregate_by_group(mice):
    groups = mice.groupby("Group ID").agg({"Mouse ID": "list",
                                            "Feature Path": "first",
                                            "Trait": "mean"})
    return groups

def parse_data():
    dotenv.load_dotenv()
    data_dir = os.getenv("DATA_DIR")
    folders = get_trait_folders(data_dir)
    microbes = pd.read_csv(os.path.join(data_dir, "microbe_abundances.csv"), index_col=0)
    for trait in folders:
        mice = os.listdir(os.path.join(data_dir, trait, "mice.csv"))
        groups = aggregate_by_group(mice)
    return

if __name__ == "__main__":
    parse_data()