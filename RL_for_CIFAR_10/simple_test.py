def a(b):
    def c():
        t = b+3
        t = t-3
        h1 = t/2
        print(h1)
    return c

f = 1759
g = a(f)
g()