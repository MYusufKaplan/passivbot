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
file_path = 'checkpoint.pkl'  # Replace with your actual file path
data = load_pickle_file(file_path)

if data:
    print("📦 Data content preview:")
    print(data)
