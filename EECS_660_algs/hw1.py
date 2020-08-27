
def q1(num1, num2):
    product = num1 * num2
    return product if product > 1000 else num1 + num2


def q2():
    pre, sum = 0, 0
    for i in range(10):
        sum = i + pre
        print(f"Current Number {i} Previous Number  {pre}  Sum:  {sum}")
        pre = i


def q3(str):
    for idx, c in enumerate(str):
        if idx % 2 == 0:
            print(c)


def q4(str, num):
    return str[num:]


def q5(num_list):
    if len(num_list) <= 1:
        return False
    return num_list[0] == num_list[-1]


def q6(num_list):
    for i in [num for num in num_list if num % 5 == 0]:
        print(i)


def q7(str, sub_str):
    return str.count(sub_str)


def q8():
    for i in range(0, 6):
        for j in range(0, i):
            print(f"{i} ", end="")
        print("\n")


def q9(num):
    print(f"original number {num}")
    num_str = str(num)
    start, end = 0, len(num_str) - 1
    while start < end:
        if num_str[start] != num_str[end]:
            return False
        start = start + 1
        end = end - 1
    return True


def q10(list1, list2):
    print(f"First List {list1}\nSecond List {list2}")
    return [i for i in list1 if i % 2 == 1] + [i for i in list2 if i % 2 == 0]


def q11(num):
    for c in str(num)[::-1]:
        print(f"{c} ", end="")


def q12(income):
    if income <= 10000:
        return 0
    elif income <= 20000:
        return (income - 10000) * 0.1
    else:
        return 10000 * 0.1 + (income - 20000) * 0.2


def q13():
    for i in range(1, 11):
        for j in range(1, 11):
            print(f"{i * j} ", end="")
        print("\n")


def q14():
    for i in range(5, 0, -1):
        for j in range(0, i, 1):
            print("* ", end="")
        print("\n")


def q15(base, exp):
    if exp < 0:
        return -1
    result = 1
    for i in range(1, exp + 1):
        result = result * base
    return result


def findA(L):
    for i in range(len(L)):
        left = i - 1
        right = i + 1
        if i == 0:
            left = i
        if i == len(L) - 1:
            right = i
        if L[left] <= L[i] >= L[right]:
            return L[i]