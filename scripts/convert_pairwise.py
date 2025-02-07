import os
import argparse
import numpy as np
import pandas as pd
from itertools import combinations

def main():
    parser = argparse.ArgumentParser(description="Create a pairwise (success - failure) comparison dataset from demos, given a single root directory.")
    
    # Only one argument: root directory
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory that must contain success_rate.txt, actions.txt, and a 'demo' folder."
    )
    
    args = parser.parse_args()
    
    # Let the user know what is expected inside root
    print("Note: This script expects the following inside the provided root directory:")
    print("  1) success_rate.txt")
    print("  2) actions.txt")
    print("  3) A folder named 'demo' containing demonstration subfolders.")
    print("  4) The output CSV will be saved to output_pairwise_dataset.csv under the same root.")
    
    # Construct the expected full paths
    success_path = os.path.join(args.root, "success_rate.txt")
    action_path = os.path.join(args.root, "actions.txt")
    demo_folder_path = os.path.join(args.root, "demo")
    dataset_output_path = os.path.join(args.root, "output_pairwise_dataset.csv")
    
    # Verify files/folders
    if not os.path.isfile(success_path):
        print(f"File not found: {success_path}")
        print("Please place 'success_rate.txt' in the root directory.")
        return
    
    if not os.path.isfile(action_path):
        print(f"File not found: {action_path}")
        print("Please place 'actions.txt' in the root directory.")
        return
    
    if not os.path.isdir(demo_folder_path):
        print(f"Folder not found: {demo_folder_path}")
        print("Please create a 'demo' folder inside the root directory.")
        return

    # Load success labels and actions
    success_labels = np.loadtxt(success_path).astype(int)
    actions = np.loadtxt(action_path).astype(int)

    # Check demo subfolders
    demo_folders = sorted(folder for folder in os.listdir(demo_folder_path) if folder.startswith("demo"))
    num_demos = len(demo_folders)
    
    # Validate dataset consistency
    if len(success_labels) != num_demos or len(actions) != num_demos:
        print(
            f"Mismatch between number of demo subfolders ({num_demos}), "
            f"success labels ({len(success_labels)}) and actions ({len(actions)})."
        )
        return

    # Collect all demos info
    all_demos = []
    for i, demo_folder_name in enumerate(demo_folders):
        demo_full_path = os.path.join(demo_folder_path, demo_folder_name)
        
        if not os.path.isdir(demo_full_path):
            print(f"Expected a folder for demo, but found: {demo_full_path}")
            return
        
        all_demos.append({
            "demo_id": demo_folder_name,
            "demo_path": demo_full_path,
            "actions": actions[i],
            "success": success_labels[i]
        })

    # Create pairwise comparison dataset
    pairwise_data = []
    for (demo1, demo2) in combinations(all_demos, 2):
        # We only compare demos with different success labels
        if demo1["success"] == demo2["success"]:
            continue

        # Label is 1 if demo1 is more successful than demo2, otherwise 0
        label = 1 if demo1["success"] > demo2["success"] else 0
        pairwise_data.append({
            "demo1_id": demo1["demo_id"],
            "demo1_path": demo1["demo_path"],
            "demo1_actions": demo1["actions"],
            "demo1_success": demo1["success"],
            "demo2_id": demo2["demo_id"],
            "demo2_path": demo2["demo_path"],
            "demo2_actions": demo2["actions"],
            "demo2_success": demo2["success"],
            "label": label
        })

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(pairwise_data)

    # Save to CSV
    df.to_csv(dataset_output_path, index=False)
    print(f"Pairwise comparison dataset created and saved to {dataset_output_path}")

if __name__ == "__main__":
    main()
