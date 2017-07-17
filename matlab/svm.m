% Jordan Ebel

function[trainSize, testError, trainError] = svm(testFM, testC, trainFM, trainC)

addpath('liblinear-2.1/matlab');

test_featureMatrix = load(testFM);
test_category = load(testC);
train_featureMatrix = load(trainFM);
train_category = load(trainC);

% train
model = svm_train(train_featureMatrix, train_category);

% test
testError = svm_test(model, test_featureMatrix, test_category);
trainError = svm_test(model, train_featureMatrix, train_category);
trainSize = size(train_featureMatrix, 2);

end
