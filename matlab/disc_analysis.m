% Jordan Ebel

testFeatureMatrix = load('testSet/test1_featureMatrix.dat');
testCategory = load('testSet/test1_category.dat');
trainFeatureMatrix = load('trainSet/train1_featureMatrix_15000.dat');
trainCategory = load('trainSet/train1_category_15000.dat');

numTestExamples = size(testFeatureMatrix, 1);
numTestFeatures = size(testFeatureMatrix, 2);
numTrainExamples = size(trainFeatureMatrix, 1);
numTrainFeatures = size(trainFeatureMatrix, 2);

%discriminant analysis:
model = fitcdiscr(trainFeatureMatrix, trainCategory);

% get train and test predictions 
test_predicted = predict(model, testFeatureMatrix);
train_predicted = predict(model, trainFeatureMatrix);

% calculate train and test errors
testError=0;
trainError=0;

for i=1:numTestExamples
  if (testCategory(i) ~= test_predicted(i))
     testError = testError + 1;
  end
end

for i=1:numTrainExamples
  if (trainCategory(i) ~= train_predicted(i))
     trainError = trainError + 1;
  end
end

testError = testError/numExamples
trainError = trainError/numExamples
