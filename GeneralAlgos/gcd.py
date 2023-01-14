def gcdbi(num1, num2):
    if num1 < num2:
        num1, num2 = num2, num1

    while num2:
        num1, num2 = num2, num1 % num2

    return num1


if __name__ == '__main__':
    print(gcdbi(60, 25))