tasks = {}

def display_tasks():
    if not tasks:
        print("No tasks available.")
    else:
        for number, task in sorted(tasks.items()):
            print(f"{number}. {task}")

def add_task():
    task = input("Add new task: ")
    number = len(tasks) + 1
    tasks[number] = task
    print(f"Task added with number {number}.")

def update_task():
    display_tasks()
    task_number = input("Enter the task number to update or type 'exit' to go back: ")
    if task_number.lower() == 'exit':
        return
    task_number = int(task_number)
    if task_number in tasks:
        new_task = input("Enter the new task: ")
        tasks[task_number] = new_task
        print("Task updated.")
    else:
        print("Invalid task number.")

def delete_task():
    display_tasks()
    task_number = input("Enter the task number to delete or type 'exit' to go back: ")
    if task_number.lower() == 'exit':
        return
    task_number = int(task_number)
    if task_number in tasks:
        del tasks[task_number]
        print(f"Task '{tasks.get(task_number, 'unknown')}' deleted.")
        reindex_tasks()
    else:
        print("Invalid task number.")

def reindex_tasks():
    global tasks
    sorted_tasks = dict(sorted(tasks.items()))
    tasks = {}
    for index, (number, task) in enumerate(sorted_tasks.items(), start=1):
        tasks[index] = task

def main():
    while True:
        option = input("Add/View/Update/Delete/exit ").strip().lower()
        if option == 'add':
            add_task()
        elif option == 'view':
            display_tasks()
        elif option == 'update':
            update_task()
        elif option == 'delete':
            delete_task()
        elif option == 'exit':
            break
        else:
            print("Invalid option. Please choose Add/View/Update/Delete/Exit.")

if __name__ == "__main__":
    main()
