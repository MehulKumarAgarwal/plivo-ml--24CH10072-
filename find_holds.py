import pandas as pd

def find_longest_holds(csv_path, top_n=5):
    try:
        # Load the dataset
        df = pd.read_csv(csv_path)
        
        # Filter strictly for 'hold' pauses
        holds = df[df['label'] == 'hold'].copy()
        
        # Calculate the duration of the pause
        holds['duration'] = holds['pause_end'] - holds['pause_start']
        
        # Sort by duration in descending order to get the longest holds at the top
        longest_holds = holds.sort_values(by='duration', ascending=False).head(top_n)
        
        # Print the results cleanly
        print(f"\nTop {top_n} longest 'hold' pauses in {csv_path}:")
        print("-" * 60)
        print(longest_holds[['turn_id', 'pause_start', 'pause_end', 'duration']].to_string(index=False))
        print("-" * 60)
        
    except FileNotFoundError:
        print(f"Error: Could not find the file at {csv_path}")
        print("Please check your file path.")

if __name__ == "__main__":
    # Check the English dataset
    print("Checking English Data...")
    find_longest_holds('../eot_data/english/labels.csv', top_n=5)
    
    # Check the Hindi dataset
    print("\nChecking Hindi Data...")
    find_longest_holds('../eot_data/hindi/labels.csv', top_n=5)
