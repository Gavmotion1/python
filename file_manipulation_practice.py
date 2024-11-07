#Opening a File (open)
file = open('example.txt', 'r')

#Closing a File (close)
file.close()

#Reading from a File (read)
content = file.read()

#Writing to a File (write)
file = open('example.txt', 'w')
file.write('Hello, world!')
file.close()

#Appending to a File (append)
file = open('example.txt', 'a')
file.write('\nAppended text.')
file.close()

#Checking Existence
import os

file_exists = os.path.exists('example.txt')
print(file_exists)  # True if file exists, False otherwise

#Deleting a File
import os

if os.path.exists('example.txt'):
    os.remove('example.txt')
    print('File deleted.')
else:
    print('File does not exist.')

# Using the with Statement
with open(file_path, mode) as file:
