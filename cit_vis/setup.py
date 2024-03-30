import numpy as np
import pandas as pd
import os
import dotenv
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
    num_traits = int(os.getenv("NUM_TRAITS", 30))
    num_microbes = int(os.getenv("NUM_MICROBES", 30))

    # Generate names for mice, traits, and microbes
    mice_names = [f"Mouse{i+1}" for i in range(num_mice)]
    traits_names = [f"Trait{i+1}" for i in range(num_traits)]
    microbes_names = [f"Microbe{i+1}" for i in range(num_microbes)]

    # Generate baseline microbe abundances
    microbe_abundances = {mouse: {microbe: np.random.normal(100, 20) for microbe in microbes_names} for mouse in mice_names}

    # Map microbes to traits based on importance
    microbe_trait_importance = {microbe: np.random.choice(traits_names, size=np.random.randint(1, 5), replace=False).tolist() for microbe in microbes_names}

    # Generate trait scores influenced by microbes
    trait_scores = pd.DataFrame(columns=traits_names, index=mice_names)
    for mouse in mice_names:
        for trait in traits_names:
            important_microbes = [microbe for microbe, traits in microbe_trait_importance.items() if trait in traits]
            microbe_influence = np.mean([microbe_abundances[mouse][microbe] for microbe in important_microbes]) if important_microbes else 100
            trait_scores.at[mouse, trait] = np.random.normal(microbe_influence, 10)

    # Convert microbe abundances to DataFrame and save
    microbe_abundances_df = pd.DataFrame(microbe_abundances).T
    trait_scores.to_csv(os.path.join(data_dir, "trait_scores.csv"))
    microbe_abundances_df.to_csv(os.path.join(data_dir, "microbe_abundances.csv"))

    # Load data
    original_trait_scores_df = pd.read_csv(os.path.join(data_dir, "trait_scores.csv"), index_col=0)
    trait_scores_df = original_trait_scores_df.copy()
    microbe_abundances_df = pd.read_csv(os.path.join(data_dir, "microbe_abundances.csv"), index_col=0)

    # Transform trait scores to categorical: low (-1), medium (0), high (1)
    def categorize_score(score, low_threshold, high_threshold):
        if score <= low_threshold:
            return -1  # Low
        elif score >= high_threshold:
            return 1   # High
        else:
            return 0   # Medium

    for trait in trait_scores_df.columns:
        low_threshold = trait_scores_df[trait].quantile(0.33)
        high_threshold = trait_scores_df[trait].quantile(0.66)
        trait_scores_df[trait] = trait_scores_df[trait].apply(lambda score: categorize_score(score, low_threshold, high_threshold))

    # Decision Tree Classifier to analyze categorical trait scores
    for trait in trait_scores_df.columns:
        os.makedirs(os.path.join(data_dir, trait), exist_ok=True)
        X = microbe_abundances_df
        y = trait_scores_df[trait]

        # Splitting the dataset (optional for this operation)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Apply the model to get predictions
        predictions = clf.predict(X)

        # Create and save a DataFrame with predictions
        results_df = pd.DataFrame({
            'Mouse ID': X.index,
            trait: original_trait_scores_df[trait][X.index],  # Real trait scores
            'Group ID': [f'Group{n}' for n in clf.apply(X)]
        })

        # Get the decision paths for the input samples
        decision_paths = clf.decision_path(X)

        # Convert the sparse matrix to a list of nodes for each sample
        paths_list = []
        for i in range(decision_paths.shape[0]):
            # Get the indices of the nodes that the sample goes through
            node_indices = decision_paths[i].indices
            # Convert the indices to a string or list and append to our list
            paths_list.append(str(node_indices))

        # Get feature names from the DataFrame
        feature_names = X.columns

        # Initialize a list to store the feature names for the paths
        feature_paths_list = []

        for i in range(decision_paths.shape[0]):
            # Get the node indices for the current sample's path
            node_indices = decision_paths[i].indices
            
            # Initialize a list to store the feature names for the current path
            path_features = []
            
            for node_index in node_indices:
                # Get the feature index for the current node
                feature_index = clf.tree_.feature[node_index]
                
                # Skip nodes that are leaves (indicated by feature_index == -2)
                if feature_index != -2:
                    # Add the feature name to the path_features list
                    feature_name = feature_names[feature_index]
                    path_features.append(feature_name)
            
            # Convert the path_features list to a string and append to feature_paths_list
            feature_paths_list.append(' -> '.join(path_features))

        # Add the feature path to the results DataFrame
        results_df['Feature Path'] = feature_paths_list

        results_df.to_csv(os.path.join(data_dir, trait, "mice.csv"), index=False)

        # Extract rules using export_text
        tree_rules = export_text(clf, feature_names=list(X.columns))

        # Save rules to a text file
        with open(os.path.join(data_dir, trait, "rules.txt"), "w") as text_file:
            text_file.write(tree_rules)

if __name__ == "__main__":
    make_data()