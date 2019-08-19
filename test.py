mytuple = ("яблоко", "банан", )
myit = iter(mytuple)
#print(type(myit))

#print(next(myit))
for i in myit:
    print(type(i), i)

for i in mytuple:
    print(type(i), i)