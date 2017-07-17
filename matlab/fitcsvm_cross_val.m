% Jordan Ebel

function [] = fitcsvm_cross_val()

folds = 4;
categories = cell(6, 1);
matrices = cell(6, 1);
results = [];

categories{1} = 'trainSet/cv/category_1.dat';
categories{2} = 'trainSet/cv/category_2.dat';
categories{3} = 'trainSet/cv/category_3.dat';
categories{4} = 'trainSet/cv/category_4.dat';
categories{5} = 'trainSet/cv/category_5.dat';
categories{6} = 'trainSet/cv/category_8.dat';
matrices{1} = 'trainSet/cv/featureMatrix_1.dat';
matrices{2} = 'trainSet/cv/featureMatrix_2.dat';
matrices{3} = 'trainSet/cv/featureMatrix_3.dat';
matrices{4} = 'trainSet/cv/featureMatrix_4.dat';
matrices{5} = 'trainSet/cv/featureMatrix_5.dat';
matrices{6} = 'trainSet/cv/featureMatrix_8.dat';
sizes = [1, 2, 3, 4, 5, 8];


for i=1:6
   
    % load data
    category = load(categories{i});
    featureMatrix = load(matrices{i});
    
    % train model
    model = fitcsvm(featureMatrix, category, 'KernelFunction', 'gaussian', 'IterationLimit', 13e3);
    
    % cross validate
    cv_model = crossval(model, 'KFold', folds);
    
    % get error
    error = kfoldLoss(cv_model);
    
    results(end+1,:) = [sizes(i), error]
    
end

hold on;
plot(results(:,1), results(:,2), '.r-');
title('3-Fold Cross Validation Error');
xlabel('Window Size');
ylabel('Error');
hold off;


