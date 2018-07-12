from __future__ import print_function

from sklearn import tree, svm, neighbors
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

train_x = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
           [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 31],
           [171, 75, 42], [181, 85, 43]]

train_y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'male']

validate_x = [[184, 84, 44], [198, 92, 48], [183, 83, 44], [166, 47, 36], [170, 60, 38],
              [172, 64, 39], [182, 80, 42], [180, 80, 43]]

validate_y = ['male', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

classifier = [["DecisionTreeClassifier", tree.DecisionTreeClassifier()], ["SVC", svm.SVC()],
              ["KNeighborsClassifier", neighbors.KNeighborsClassifier()],
              ["GaussianNB", GaussianNB()]]

metrics = (None, 0)

for name, clf in classifier:
    clf = clf.fit(train_x, train_y)

    prediction = clf.predict(validate_x)
    accuracy = accuracy_score(validate_y, prediction)

    print('Accuracy for %s is %f' % (name, accuracy))

    if metrics[1] < accuracy:
        metrics = (name, accuracy)

print('\nBest accuracy for %s with %f' % metrics)
