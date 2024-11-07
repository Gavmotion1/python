contacts = []
while True:
    prompt = input("Do you want to add more contacts? ")


    if prompt.lower() == "yes":
        first_name = input("first name: ")
        last_name = input("last name: ")
        phone_number = input("phone number: ")

        contacts.append((first_name, last_name, phone_number))

        contacts.sort(key=lambda contact: contact[0])
    elif prompt.lower() == "no":
        break
    else:
        print("please enter yes or no " )
print(contacts)
