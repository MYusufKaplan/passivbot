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
file_path = 'evaluator.pkl'  # Replace with your actual file path
data = load_pickle_file(file_path)

if data:
    print("📦 Data content preview:")
    data['btc_usd_shared_memory_files'] = {"combined": '/home/myusuf/Projects/passivbot/btc_usd_tempFile'}
    # sample = data['population'][0]
    # data['population']
    # for idx in range(0,len(data["population"])):
    #     data['population'].pop()
    # data['population'].append(sample)
    with open(file_path, "wb") as f:
                pickle.dump(data, f)
    # print(data)
