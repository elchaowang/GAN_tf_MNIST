import logging

def use_logging(func):
    print("%s is running " % func.__name__)
    return func

@use_logging
def foo():
    print('i am foo')

foo()