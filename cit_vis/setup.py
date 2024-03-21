import numpy as np
import pandas as pd
import os
import dotenv
import json
import re
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

def make_data():
    # Load environment variables
    dotenv.load_dotenv()

    # Ensure the data directory exists
    data_dir = os.getenv("DATA_DIR")
    os.makedirs(data_dir, exist_ok=True)

    # Initialize parameters
    num_mice = int(os.getenv("NUM_MICE", 50))
    num_assays = int(os.getenv("NUM_ASSAYS", 30))
    num_genera = int(os.getenv("NUM_GENERA", 30))

    # Generate names for mice, assays, and genera
    mice_names = [f"Mouse_{i+1}" for i in range(num_mice)]
    assays_names = [f"Assay_{i+1}" for i in range(num_assays)]
    genera_names = [f"Genus_{i+1}" for i in range(num_genera)]

    # Generate baseline genus abundances
    genus_abundances = {mouse: {genus: np.random.normal(100, 20) for genus in genera_names} for mouse in mice_names}

    # Map genera to assays based on importance
    genus_assay_importance = {genus: np.random.choice(assays_names, size=np.random.randint(1, 5), replace=False).tolist() for genus in genera_names}

    # Generate assay scores influenced by genera
    assay_scores = pd.DataFrame(columns=assays_names, index=mice_names)
    for mouse in mice_names:
        for assay in assays_names:
            important_genera = [genus for genus, assays in genus_assay_importance.items() if assay in assays]
            genus_influence = np.mean([genus_abundances[mouse][genus] for genus in important_genera]) if important_genera else 100
            assay_scores.at[mouse, assay] = np.random.normal(genus_influence, 10)

    # Convert genus abundances to DataFrame and save
    genus_abundances_df = pd.DataFrame(genus_abundances).T
    assay_scores.to_csv(os.path.join(data_dir, "assay_scores.csv"))
    genus_abundances_df.to_csv(os.path.join(data_dir, "genus_abundances.csv"))

    # Load data
    original_assay_scores_df = pd.read_csv(os.path.join(data_dir, "assay_scores.csv"), index_col=0)
    assay_scores_df = original_assay_scores_df.copy()
    genus_abundances_df = pd.read_csv(os.path.join(data_dir, "genus_abundances.csv"), index_col=0)

    # Transform assay scores to categorical: low (-1), medium (0), high (1)
    def categorize_score(score, low_threshold, high_threshold):
        if score <= low_threshold:
            return -1  # Low
        elif score >= high_threshold:
            return 1   # High
        else:
            return 0   # Medium

    for assay in assay_scores_df.columns:
        low_threshold = assay_scores_df[assay].quantile(0.33)
        high_threshold = assay_scores_df[assay].quantile(0.66)
        assay_scores_df[assay] = assay_scores_df[assay].apply(lambda score: categorize_score(score, low_threshold, high_threshold))

    relevant_genera = {}
    relevant_assays = {}

    # Decision Tree Classifier to analyze categorical assay scores
    for assay in assay_scores_df.columns:
        try:
            os.mkdir(f"data/{assay}")
        except FileExistsError:
            pass
        X = genus_abundances_df
        y = assay_scores_df[assay]

        # Splitting the dataset (optional for this operation)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Apply the model to get predictions
        predictions = clf.predict(X)

        def get_decision_path(mouse):
            feature_names = [X.columns[i] for i in clf.decision_path(X.loc[[mouse]]).indices]
            return feature_names

        def format_decision_path(mouse):
            end_node = clf.apply(X.loc[[mouse]])[0]
            main_path = ('>').join(get_decision_path(mouse))
            full_path = main_path + f'>Node_{end_node}'
            return full_path

        # Create and save a DataFrame with predictions
        results_df = pd.DataFrame({
            'Mouse_ID': X.index,
            'Assay': original_assay_scores_df[assay][X.index],  # Real assay scores
            'Node_ID': [f'Node_{n}' for n in clf.apply(X)],  # Node where the prediction was made
            'Path': [format_decision_path(mouse) for mouse in X.index],
        })
        results_df.to_csv(f"data/{assay}/mice.csv", index=False)

        # Extract rules using export_text
        tree_rules = export_text(clf, feature_names=list(X.columns))

        # Save rules to a text file
        with open(f"data/{assay}/rules.txt", "w") as text_file:
            text_file.write(tree_rules)

        genera = []
        for line in tree_rules.split("\n"):
            if "Genus_" in line:
                # Use regular expression to extract the genus ID
                genus = re.findall(r'Genus_\d+', line)[0]
                if genus not in genera:
                    genera.append(genus)

        # Store relevant genera:
        relevant_genera[assay] = genera

        # Store relevant assays:
        for genus in genera:
            if genus in relevant_assays:
                relevant_assays[genus].append(assay)
            else:
                relevant_assays[genus] = [assay]

    with open("data/relevant_genera.json", "w") as json_file:
        json.dump(relevant_genera, json_file)

    with open("data/relevant_assays.json", "w") as json_file:
        json.dump(relevant_assays, json_file)

if __name__ == "__main__":
    make_data()