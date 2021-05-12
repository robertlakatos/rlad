from newsspace200 import Newsspace200

ns200 = Newsspace200("data")
print(ns200.get_labels())
train = ns200.get_train() 
print(train)
test = ns200.get_test()
print(test)