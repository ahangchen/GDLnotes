# Numerical stability

a = 1000000000
for i in xrange(1000000):
    a += 1e-6
print(a - 1000000000)

a = 1
for i in xrange(1000000):
    a += 1e-6
print a - 1