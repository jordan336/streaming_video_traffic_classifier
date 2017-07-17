% Jordan Ebel

function [ model ] = svm_train( featureMatrix, category )

    numExamples = size(featureMatrix, 1);
    numFeatures = size(featureMatrix, 2);

    % convert to -1/1 convention
    for i=1:numExamples
        if category(i) == 0
            category(i) = -1.0;
        else
            category(i) = 1.0;
        end
    end

    %trainLabels = transpose(category);
    trainLabels = category;

    featureMatrix = sparse(featureMatrix);

    % overcome "sparse matrix" error by writing then reading from file
    libsvmwrite('localSvm', trainLabels, featureMatrix);
    [label_vector, instance_matrix] = libsvmread('localSvm');

    % train the model
    model = train(label_vector, instance_matrix);
 end

