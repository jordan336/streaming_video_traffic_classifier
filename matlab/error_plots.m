% Jordan Ebel

function [] = error_plots(testDir, trainDir)

i = 0; y=0;
testCatFile = strcat(testDir, '/category.dat');
testFmFile = strcat(testDir, '/featureMatrix.dat');

trainSizes = [];
nb_testErrors = [];
nb_trainErrors = [];
nb_testErrors_sorted = [];
nb_trainErrors_sorted = [];
lr_testErrors = [];
lr_trainErrors = [];
lr_testErrors_sorted = [];
lr_trainErrors_sorted = [];
sv_testErrors = [];
sv_trainErrors = [];
sv_testErrors_sorted = [];
sv_trainErrors_sorted = [];
sv_trainSizes = [];

trainCatFiles = dir(strcat(trainDir, '/category_*.dat'));

% run algorithms for every training size in training directory
for file = trainCatFiles'
    i = i + 1;
    
    trainFmFile = strcat(trainDir, '/featureMatrix', file.name(9:end));
    trainCatFile = strcat(trainDir, '/', file.name);
    
    [trainSize, testError, trainError] = naive_bayes(testFmFile, testCatFile, trainFmFile, trainCatFile);
    trainSizes(end+1, :) = [trainSize,i];
    nb_testErrors(end+1) = testError;
    nb_trainErrors(end+1) = trainError;
    
    [trainSize, testError, trainError] = logistic_regression(testFmFile, testCatFile, trainFmFile, trainCatFile);
    lr_testErrors(end+1) = testError;
    lr_trainErrors(end+1) = trainError;
    
    %[trainSize, testError, trainError] = svm(testFmFile, testCatFile, trainFmFile, trainCatFile);
    
    if (trainSize < 90000)
        y = y + 1;
        [trainSize, testError, trainError] = fitcsvm_run(testFmFile, testCatFile, trainFmFile, trainCatFile);
        sv_testErrors(end+1) = testError;
        sv_trainErrors(end+1) = trainError;
        sv_trainSizes(end+1,:) = [trainSize, y];
    end
end

% sort by number of training examples
sorted = sortrows(trainSizes, 1);
sv_sorted = sortrows(sv_trainSizes, 1);

% reorder error arrays according to number of training examples
for x = 1:size(sorted, 1)
    nb_testErrors_sorted(x) = nb_testErrors(sorted(x, 2));
    nb_trainErrors_sorted(x) = nb_trainErrors(sorted(x, 2));
    lr_testErrors_sorted(x) = lr_testErrors(sorted(x, 2));
    lr_trainErrors_sorted(x) = lr_trainErrors(sorted(x, 2));
end
for x = 1:size(sv_sorted, 1)
    sv_testErrors_sorted(x) = sv_testErrors(sv_sorted(x, 2));
    sv_trainErrors_sorted(x) = sv_trainErrors(sv_sorted(x, 2));
end

% plot test and training errors on one plot
hold on;
plot(sorted(:,1), nb_testErrors_sorted, '.r-');
plot(sorted(:,1), nb_trainErrors_sorted, '.r--');
plot(sorted(:,1), lr_testErrors_sorted, '.b-');
plot(sorted(:,1), lr_trainErrors_sorted, '.b--');
plot(sv_sorted(:,1), sv_testErrors_sorted, '.g-');
plot(sv_sorted(:,1), sv_trainErrors_sorted, '.g--');
title('Test and Train Errors');
xlabel('Num. Training Examples');
ylabel('Error');
legend('Naive Bayes Test', 'Naive Bayes Train', 'Logistic Regression Test', 'Logistic Regression Train', 'SVM Test', 'SVM Train');
hold off;

