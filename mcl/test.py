weight =[1, 2, 3]
probs = [sum(weight[:i+1]) for i in range(3)]
print (probs)
