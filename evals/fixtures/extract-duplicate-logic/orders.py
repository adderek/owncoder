def total_with_tax_us(price, quantity):
    subtotal = price * quantity
    tax = subtotal * 0.08
    return round(subtotal + tax, 2)


def total_with_tax_ca(price, quantity):
    subtotal = price * quantity
    tax = subtotal * 0.13
    return round(subtotal + tax, 2)
