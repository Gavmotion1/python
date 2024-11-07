def calculate_percentile(placement, total_students):
    if placement < 1 or placement > total_students:
        raise ValueError("Placement must be between 1 and the total number of students.")
    percentile = (1 - (placement / total_students)) * 100
    return percentile

placement = int(input("whats placement "))
total_students = int(input("total students "))
print(f"The percentile rank is: {calculate_percentile(placement, total_students):.2f}%")
