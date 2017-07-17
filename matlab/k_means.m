% Jordan Ebel

function[trainError] = k_means(trainFM, trainC)

trainFeatureMatrix = load(trainFM);
trainCategory = load(trainC);

numTrainingExamples = size(trainFeatureMatrix, 1);
positive_group1 = [];
positive_group2 = [];
negative_group1 = [];
negative_group2 = [];

% k means
[cluster_index, ~] = kmeans(trainFeatureMatrix, 2);

% PCA
[~, score, ~, ~, ~] = pca(trainFeatureMatrix, 'VariableWeights', 'variance');

% group points for scatter plot
for i = 1:numTrainingExamples   
    if trainCategory(i) == 1
        if cluster_index(i) == 1
            positive_group1(end+1,:) = [score(i, 1); score(i, 2)];
        else
            positive_group2(end+1,:) = [score(i, 1); score(i, 2)];
        end
    else
        if cluster_index(i) == 1
            negative_group1(end+1,:) = [score(i, 1); score(i, 2)];
        else
            negative_group2(end+1,:) = [score(i, 1); score(i, 2)];
        end
    end
end

% scatter plot of points in PCA space
figure
hold on
s1 = scatter(positive_group1(:,1), positive_group1(:,2), 'red', 'o');
s2 = scatter(positive_group2(:,1), positive_group2(:,2), 'red', 'x')
s3 = scatter(negative_group1(:,1), negative_group1(:,2), 'blue', 'o');
s4 = scatter(negative_group2(:,1), negative_group2(:,2), 'blue', 'x');
title('2-Means Clustering of Dataset in Principle Component Space');
xlabel('Component 1');
ylabel('Component 2');
leg = legend([s1, s2, s3, s4], {'Positive Example, Cluster 1', 'Positive Example, Cluster 2', 'Negative Example, Cluster 1', 'Negative Example, Cluster 2'});
hold off

% calculate train and test errors
trainError=0;

% calculate train error
for i=1:numTrainingExamples
  if (trainCategory(i) == 0)
    if (cluster_index(i) == 1)
        trainError = trainError + 1;
    end
  end
  if (trainCategory(i) == 1)
    if (cluster_index(i) == 2)
        trainError = trainError + 1;
    end
  end
end

trainError = trainError/numTrainingExamples;

end
