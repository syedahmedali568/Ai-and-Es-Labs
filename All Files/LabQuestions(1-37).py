#!/usr/bin/env python
# coding: utf-8

# In[2]:


"*" * 10


# In[2]:


multiline_text = """This is a
multiline string
that spans multiple lines."""


# In[4]:


def example_function():
    """This is a docstring for the example_function.
    It provides information about the function's purpose and behavior."""
    pass


# In[ ]:


text = """He said, "It's a sunny day!" """


# In[16]:


name = "John Smith"
[name[1], name[-2], name[1:-1], len(name)]
print(name[1])
print(name[-2])
print(name[1:-1])
print(len(name))


# In[18]:


name.title()


# In[24]:


name = "   john smith   "
stripped_name = name.strip()
stripped_name


# In[30]:


name = "John smith"
name.find('Smith')


# In[16]:


name = "John smith"
name.replace("J","k")


# In[18]:


"John" in name


# In[24]:


10**3


# In[26]:


x = 1
x+=2
x


# In[32]:


round(3.14159, 2)


# In[34]:


float(1)


# In[36]:


10 == "10"


# In[38]:


"bag"> "apple"


# In[51]:


not(True or False)


# In[61]:


print(list(range(1,10,2)))


# In[65]:


def maximum(a, b):
    if a > b:
        return a
    else:
        return b
result = maximum(5, 10)
print(result)


# In[67]:


def check_num(num):
    if num % 3 == 0 and num % 5 == 0:
        return "Divisible by both"
    elif num % 3 == 0:
        return "Divisible by 3"
    elif num % 5 == 0:
        return "Divisible by 5"
    else:
        return num


# In[71]:


print(check_num(15)) 
print(check_num(9))  
print(check_num(10)) 
print(check_num(7))   


# In[91]:


def showNumbers(limit):
    return list(range(0,limit+1))

showNumbers(5)


# In[103]:


import math

def calculate_areas(radius):
    #for Circle
    area_circle = math.pi * (radius ** 2)
    #for Square
    side_length_square = 2 * radius
    area_square = side_length_square ** 2
    #for Rectangle
    length_rectangle = 2 * radius
    width_rectangle = radius  
    area_rectangle = length_rectangle * width_rectangle
    #for Triangle
    base_triangle = 2 * radius
    height_triangle = radius
    area_triangle = 0.5 * base_triangle * height_triangle
    return area_circle, area_square, area_rectangle, area_triangle
    
radius = float(input("Enter the radius of the circle: "))

area_circle, area_square, area_rectangle, area_triangle = calculate_areas(radius)

print(f"Area of the circle: {area_circle:.2f}")
print(f"Area of the square: {area_square:.2f}")
print(f"Area of the rectangle: {area_rectangle:.2f}")
print(f"Area of the triangle: {area_triangle:.2f}")    


# In[113]:


def sum_of_multiples(limit):
    total_sum = 0
    for number in range(limit + 1):
        if number % 3 == 0 or number % 5 == 0:
            total_sum += number
    return total_sum

# Example usage
limit = int(input("Enter the limit: "))
result = sum_of_multiples(limit)
print(f"The sum of multiples of 3 and 5 up to {limit} is {result}")


# In[115]:


def show_stars(rows):
    for i in range(1, rows + 1):
        print('*' * i)
show_stars(5)


# In[117]:


def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def print_primes(limit):
    for num in range(2, limit + 1):
        if is_prime(num):
            print(num)
print_primes(20)


# In[2]:


first_name = input("Enter your first name: ")
last_name = input("Enter your last name: ")

print(last_name + " " + first_name)


# In[6]:


number = int(input("Enter a number: "))
if number % 2 == 0:
    print(f"{number} is an even number.")
else:
    print(f"{number} is an odd number.")


# In[8]:


n = input("Enter an integer: ")
result = int(n) + int(n*2) + int(n*3)
print(f"The result of n + nn + nnn is: {result}")


# In[12]:


letter = input("Enter a letter: ").lower()
if letter in 'aeiou':
    print(f"{letter} is a vowel.")
else:
    print(f"{letter} is not a vowel.")


# In[16]:


def sum_three_integers(a, b, c):
    if a == b or b == c or a == c:
        return 0
    else:
        return a + b + c

num1 = int(input("Enter the first integer: "))
num2 = int(input("Enter the second integer: "))
num3 = int(input("Enter the third integer: "))

result = sum_three_integers(num1, num2, num3)
print(f"The result is: {result}")


# In[24]:


def check_values(a, b):
    if a == b:
        return True
    elif a + b == 5:
        return True
    elif abs(a - b) == 5:
        return True
    else:
        return False
num1 = int(input("Enter the first integer: "))
num2 = int(input("Enter the second integer: "))

result = check_values(num1, num2)
print(f"The result is: {result}")


# In[26]:


def calculate_expression(x, y, z):
    result = (x + y) ** z
    return result
x = 4
y = 3
z = 2
output = calculate_expression(x, y, z)
print(f"({x} + {y}) ^ {z} = {output}")


# In[ ]:




