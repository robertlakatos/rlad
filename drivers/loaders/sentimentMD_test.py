from sentimentMD import SentimentMD

sMD = SentimentMD("data")
print(sMD.get_labels())
train = sMD.get_train() 
print(train)
test = sMD.get_test()
print(test)