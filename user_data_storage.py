# user_data_storage.py
import json
import os

class Credentials:
    def __init__(self, username, password, is_admin=False):
        self.username = username
        self.password = password # WARNING: Storing plaintext passwords is insecure
        self.is_admin = is_admin

    def to_dict(self):
        return {
            'username': self.username,
            'password': self.password,
            'is_admin': self.is_admin
        }

def create_folder_if_not_exist(folder):
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
            print(f"Info: Created folder '{folder}'")
        except Exception as e:
            print(f"Error: Could not create folder '{folder}': {e}")


def read_credentials(file_path):
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
            if isinstance(data, dict):
                return {k: Credentials(**v) for k, v in data.items()}
            else:
                print(f"Warning: Credentials file '{file_path}' did not contain a valid dictionary.")
                return {}
    except FileNotFoundError:
        print(f"Info: Credentials file '{file_path}' not found. A new one will be created with a default admin.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from credentials file '{file_path}'. File might be corrupted.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred while reading credentials file '{file_path}': {e}")
        return {}


def write_credentials(file_path, credentials_dict):
    data = {k: v.to_dict() for k, v in credentials_dict.items()}
    try:
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        print(f"An unexpected error occurred while writing credentials to '{file_path}': {e}")

# File storage location
storage_folder = "tmp_data"
storage_file = os.path.join(storage_folder, "user_credentials.json")

# Ensure the folder exists before trying to read/write
create_folder_if_not_exist(storage_folder)

# Load existing user data
credentials = read_credentials(storage_file)

# If 'admin' user doesn't exist (e.g., first run or corrupted file), initialize it
if 'admin' not in credentials:
    print("Info: Default 'admin' account not found. Initializing with password 'admin123'.")
    admin_username = "admin"
    admin_password = "admin123" # WARNING: Highly insecure default password!
    admin_user = Credentials(admin_username, admin_password, True)
    credentials[admin_username] = admin_user
    write_credentials(storage_file, credentials)