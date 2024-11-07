import json
import os

# File to store passwords
password_file = 'passwords.json'

# Load existing passwords from file
def load_passwords():
    if os.path.exists(password_file):
        with open(password_file, 'r') as file:
            return json.load(file)
    return {}

# Save passwords to file
def save_passwords(passwords):
    with open(password_file, 'w') as file:
        json.dump(passwords, file)

# Initialize passwords from file
passwords = load_passwords()

def password_manager():
    while True:
        user_choice = input('\nInput values:\n1. Add Password\n2. View Passwords\n3. Edit Password\n4. Delete Password\n5. Exit\n').lower()
        
        if user_choice == '1':
            website = input('Input website, or app name:\n')
            password = input('Input password:\n')
            passwords[website] = password
            save_passwords(passwords)  # Save after adding
            print(f"Password for {website} added successfully.")
        
        elif user_choice == '2':
            if passwords:
                for site, pwd in passwords.items():
                    print(f"Website/App: {site}, Password: {pwd}")
            else:
                print("No passwords stored.")
        
        elif user_choice == '3':
            website = input('Input website, or app name to edit:\n')
            if website in passwords:
                new_password = input('Input new password:\n')
                passwords[website] = new_password
                save_passwords(passwords)  # Save after editing
                print(f"Password for {website} updated successfully.")
            else:
                print("Website/App not found.")
        
        elif user_choice == '4':
            website = input('Input website, or app name to delete:\n')
            if website in passwords:
                del passwords[website]
                save_passwords(passwords)  # Save after deleting
                print(f"Password for {website} deleted successfully.")
            else:
                print("Website/App not found.")
        
        elif user_choice == '5':
            print("Exiting the password manager.")
            break
        
        else:
            print("Invalid choice. Please try again.")

password_manager()
