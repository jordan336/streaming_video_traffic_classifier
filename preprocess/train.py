# Jordan Ebel

from sklearn import svm
from sklearn.externals import joblib
import re

category = []
featureMatrix = []

cat_file = open("category.dat", "r")
for line in cat_file:
    category.append(int(line))

fm_file = open("featureMatrix.dat", "r")
for line in fm_file:
    nums = [float(n) for n in line.split()]
    featureMatrix.append(nums)

print "Fitting model"

model = svm.SVC(cache_size=1000, class_weight='balanced')
model.fit(featureMatrix, category)

print "Done fitting model"
print model

joblib.dump(model, 'model/model.pkl')

