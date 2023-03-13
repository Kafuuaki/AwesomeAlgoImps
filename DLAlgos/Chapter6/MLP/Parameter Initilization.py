import torch
from torch import nn
from torch import functional as F

# def init_nomal(module):



if __name__ == "__main__":


    def adecorator(afunc):
        def wrapper():
            print('a')
            afunc()
            print('b')

        return wrapper

    @adecorator
    def usingd():
        print("usingd")

    usingd()

    # a = usingd()
    # print(usingd == None)
    # usingd()
    # print("a")

# def hi():
#     return "hi yasoob!"
#
#
# def doSomethingBeforeHi(func):
#     print("I am doing some boring work before executing hi()")
#     print(func())