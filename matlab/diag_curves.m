% Jordan Ebel

function [] = diag_curves(testDir, trainFM, trainC)

testCatFile = strcat(testDir, '/category.dat');
testFmFile = strcat(testDir, '/featureMatrix.dat');
trainFmFile = strcat(trainFM);
trainCatFile = strcat(trainC);
    
[~, ~, ~, xRocNb, yRocNb, rocAucNb, xRpNb, yRpNb, rpAucNb, tpNb, tnNb, fpNb, fnNb] = naive_bayes(testFmFile, testCatFile, trainFmFile, trainCatFile);
    
[~, ~, ~, xRocLr, yRocLr, rocAucLr, xRpLr, yRpLr, rpAucLr, tpLr, tnLr, fpLr, fnLr] = logistic_regression(testFmFile, testCatFile, trainFmFile, trainCatFile);
    
[~, ~, ~, xRocSv, yRocSv, rocAucSv, xRpSv, yRpSv, rpAucSv, tpSv, tnSv, fpSv, fnSv] = fitcsvm_run(testFmFile, testCatFile, trainFmFile, trainCatFile);

rocAucNb
rpAucNb
confusionNb = [tnNb, fnNb; fpNb, tpNb]

rocAucLr
rpAucLr
confusionLr = [tnLr, fnLr; fpLr, tpLr]

rocAucSv
rpAucSv
confusionSv = [tnSv, fnSv; fpSv, tpSv]

% plot ROC curves
figure
hold on;
plot(xRocNb, yRocNb, 'r-');
plot(xRocLr, yRocLr, 'b-');
plot(xRocSv, yRocSv, 'g-');
title('ROC Curves');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
legend('Naive Bayes', 'Logistic Regression', 'SVM');
hold off;

% plot recall/precision curves
figure
hold on;
plot(xRpNb, yRpNb, 'r-');
plot(xRpLr, yRpLr, 'b-');
plot(xRpSv, yRpSv, 'g-');
title('Recall-Precision Curves');
xlabel('Recall');
ylabel('Precision');
legend('Naive Bayes', 'Logistic Regression', 'SVM');
hold off;
