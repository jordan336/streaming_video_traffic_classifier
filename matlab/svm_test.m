% Jordan Ebel

function [ error ] = svm_test( model, featureMatrix, category )

    numExamples = size(featureMatrix, 1);
    numFeatures = size(featureMatrix, 2);

    output = zeros(numExamples, 1);

    % vector of the real labels so predict can report its accuracy
    testActualCategories = zeros(numExamples, 1);

    % convert to -1/1 convention
    for i=1:numExamples
        if category(i) == 0
            testActualCategories(i) = -1.0;
        else
            testActualCategories(i) = 1.0;
        end
    end

    featureMatrix = sparse(featureMatrix);

    % overcome "sparse matrix" error by writing then reading from file
    libsvmwrite('localSvmTest', testActualCategories, featureMatrix);
    [label_vector, instance_matrix] = libsvmread('localSvmTest');

    % make prediction based on model
    output = predict(label_vector, instance_matrix, model);

    % convert to 0/1 convention
    for i=1:numExamples
        if output(i) == -1.0
            output(i) = 0;
        else
            output(i) = 1;
        end
    end


    % Compute the error on the test set
    error_count=0;
    for i=1:numExamples
      if (category(i) ~= output(i))
        error_count=error_count+1;
      end
    end

    %Print out the classification error on the test set
    error = error_count/numExamples;
end
