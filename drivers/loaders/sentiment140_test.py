from sentiment140 import Sentiment140

s140 = Sentiment140("data")
print(s140.get_labels())
train = s140.get_train() 
print(train)
test = s140.get_test()
print(test)