def create_sorted_data():
    data = [
        "Apple", "Banana", "Cherry", "Avocado", "Blueberry",
        "Coconut", "Blackberry", "Apricot", "Carrot", "Beetroot",
        "Almond", "Avocado", "Cucumber", "Artichoke", "Cabbage"
    ]
    data.sort()  # Sort data alphabetically
    return data

def build_index(data):
    index = {}
    for item in data:
        first_letter = item[0].upper()
        if first_letter not in index:
            index[first_letter] = []
        index[first_letter].append(item)
    return index

def get_items_by_letter(index, letter):
    letter = letter.upper()
    if letter in index:
        return index[letter]
    else:
        return []

def main():
    data = create_sorted_data()
    index = build_index(data)
    
    while True:
        letter = input("Enter a letter to see items starting with it (or 'quit' to exit): ").strip()
        if letter.lower() == 'quit':
            break
        if len(letter) == 1 and letter.isalpha():
            items = get_items_by_letter(index, letter)
            if items:
                print(f"Items starting with '{letter}':")
                for item in items:
                    print(item)
            else:
                print(f"No items start with '{letter}'.")
        else:
            print("Invalid input. Please enter a single letter.")
    
if __name__ == "__main__":
    main()
