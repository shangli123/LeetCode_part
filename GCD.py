import math


def isprime(t):
    for i in range(2, math.ceil(t / 2) + 1):
        if t % i == 0:
            return False
    return True

# O(n^2)
def gcd_slow(k):
    gcdset = []
    counter = 0
    for n in range(2, math.ceil(k / 2) + 1):
        if isprime(n):
            if k % n == 0:
                gcdset = gcdset + [n]
                counter = counter + 1
                print(counter)
                continue
    return gcdset


def dividprime(intermediate, initprime, gcdset, flag):
    if intermediate == initprime:
        flag = 1
    if intermediate % initprime == 0:
        gcdset = gcdset + [initprime]
        intermediate = intermediate / initprime
    else:
        initprime = initprime + 1
    return intermediate, initprime, gcdset, flag

# O(n)
def gcd_med(k):
    initp = 2
    interm = k
    flag = 0
    gcdset = []
    while flag == 0:
        interm, initp, gcdset, flag = dividprime(interm, initp, gcdset, flag)
    gcdset = set(gcdset)
    gcdset = list(gcdset)
    return gcdset



var1 = int(input('please input: '))

print(gcd_med(var1))
print(gcd_slow(var1))
