products = [
    ['Laptop', 999.99, 5],
    ['Mouse', 19.99, 50],
    ['Keyboard', 49.99, 30],
    ['Monitor', 199.99, 20],
    ['Printer', 149.99, 10],
    ['Desk', 299.99, 7],
    ['Chair', 89.99, 15],
]
product_template = 'Product: {}, Price per Unit: ${:.2f}, Quantity: {}, Total Value: ${:.2f}'
for product,price_per_Unit,quantity in products:
    total_value = price_per_Unit * quantity
    
    product_summary = product_template.format(product, price_per_Unit, quantity, total_value)
    print(product_summary)
