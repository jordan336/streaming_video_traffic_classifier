% Jordan Ebel

function [trainSize, testError, trainError, rocX, rocY, rocAuc, rpX, rpY, rpAuc, truePositiveRate, trueNegativeRate, falsePositiveRate, falseNegativeRate] = logistic_regression(testFM, testC, trainFM, trainC)

testFeatureMatrix = load(testFM);
testCategory = load(testC);
trainFeatureMatrix = load(trainFM);
trainCategory = load(trainC);

numTestExamples = size(testFeatureMatrix, 1);
numTestFeatures = size(testFeatureMatrix, 2);
numTrainExamples = size(trainFeatureMatrix, 1);
numTrainFeatures = size(trainFeatureMatrix, 2);

% train model
model = fitglm(trainFeatureMatrix, trainCategory, 'Distribution', 'binomial');

% get train and test predictions 
test_predicted = predict(model, testFeatureMatrix);
train_predicted = predict(model, trainFeatureMatrix);

% calculate train and test errors, and false / true rates
testError=0;
trainError=0;
testFp = 0;
testFn = 0;
testTp = 0;
testTn = 0;
totalP = 0;
totalN = 0;

% calculate test error
for i=1:numTestExamples
  if(test_predicted(i) > 0.5)
     test_predicted(i) = 1;
  else
     test_predicted(i) = 0;
  end
    
  if (testCategory(i) == 0)
      totalN = totalN + 1;
      
      % true negative
      if (test_predicted(i) == 0)
          testTn = testTn + 1;
      % false positive
      else
          testFp = testFp + 1;
          testError = testError + 1;
      end
  else
      totalP = totalP + 1;
      
      % false negative
      if (test_predicted(i) == 0)
          testFn = testFn + 1;
          testError = testError + 1;
      % true positive
      else
          testTp = testTp + 1;
      end
  end
  
end

% calculate train error
for i=1:numTrainExamples
  if (train_predicted(i) > 0.5)
     train_predicted(i) = 1;
  else
      train_predicted(i) = 0;
  end
  
  if (trainCategory(i) ~= train_predicted(i))
     trainError = trainError + 1;
  end
end

testError = testError/numTestExamples;
trainError = trainError/numTrainExamples;
trainSize = numTrainExamples;

% calculate true / false error rates
truePositiveRate = testTp / totalP;
trueNegativeRate = testTn / totalN;
falsePositiveRate = testFp / totalN;
falseNegativeRate = testFn / totalP;

% calculate ROC values
scores = model.Fitted.Probability;
[rocX, rocY, ~, rocAuc] = perfcurve(trainCategory, scores, 1); 

% calculate recall / precision values
[rpX, rpY, ~, rpAuc] = perfcurve(trainCategory, scores, 1, 'xCrit', 'reca', 'yCrit', 'prec');

end
