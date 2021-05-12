from imdb import IMDB

iamdb = IMDB("data")
print(iamdb.get_labels())
train = iamdb.get_train()
print(train)
test = iamdb.get_test()
print(test)