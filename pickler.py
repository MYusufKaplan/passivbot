import pickle

def load_pickle_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            print("✅ Pickle file loaded successfully!")
            return data
    except FileNotFoundError:
        print("❌ File not found. Please check the path.")
    except pickle.UnpicklingError:
        print("❌ Error unpickling the file. Is this a valid pickle file?")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

# Example usage
import os

file_path = 'pso_history_data.pkl'  # Replace with your actual file path
data = load_pickle_file(file_path)
DESIRED_POP_SIZE = 3000

if data:
    print("📦 Starting pickle sizer...")
    print(f"🧠 Current population size: {len(data['population'])}")
    print(f"🎯 Desired population size: {DESIRED_POP_SIZE}")

    sample = data['population'][0]

    while len(data["population"]) != DESIRED_POP_SIZE:
        if len(data["population"]) > DESIRED_POP_SIZE:
            removed = data['population'].pop()
            print(f"➖ Removed one individual. New size: {len(data['population'])}")
        else:
            data["population"].append(sample)
            print(f"➕ Added one individual. New size: {len(data['population'])}")

    print("✅ Finished pickle sizer!")
    print(f"📊 Final population size: {len(data['population'])}")
else:
    print("❌ Failed to load data from pickle file.")

# data['population'].append(sample)
with open(file_path, "wb") as f:
    pickle.dump(data, f)
# print(data)
