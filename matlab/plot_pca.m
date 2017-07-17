% Jordan Ebel

function [] = plot_pca(trainFM, trainC)

trainFeatureMatrix = load(trainFM);
trainCategory = load(trainC);

varlabels = {'inter_packet time', 'packet size', 'ip length', 'offset', 'protocol', 'TTL', 'mean IA time', 'var IA time', 'mean ip length', 'var ip length', 'mean TTL', 'var TTL', 'mean protocol', 'var protocol'};
%varlabels = {'inter_packet time', 'packet size', 'ip length', 'offset', 'protocol', 'TTL'};
numTrainingExamples = size(trainCategory, 1);

positives = [];
negatives = [];

% PCA
[coeff, score, latent, tsquared, explained] = pca(trainFeatureMatrix, 'VariableWeights', 'variance');

% collect orthonormal coefficients 
coefforth = inv(diag(std(trainFeatureMatrix))) * coeff;

% output variance explained matrix
explained

% group points for scatter plot
for i = 1:numTrainingExamples   
    if trainCategory(i) == 1
        positives(end+1,:) = [score(i, 1); score(i, 2)];
    else
        negatives(end+1,:) = [score(i, 1); score(i, 2)];
    end
end

% scatter plot of points in PCA space
figure
hold on
scatter(positives(:,1), positives(:,2), 'red', 'o');
scatter(negatives(:,1), negatives(:,2), 'blue', 'square');
title('Representation of Dataset in Principle Component Space');
xlabel('Component 1');
ylabel('Component 2');
legend('Positive Examples', 'Negative Examples');
hold off

% biplot of points in PCA space
figure
bip = biplot(coefforth(:, 1:2), 'scores', score(:,1:2), 'varlabels', varlabels, 'ObsLabels', num2str((1:size(score,1))'));
title('Dataset and Features in Principle Component Space');

% adjust color of points in biplot according to label
for i = 1:length(bip)
    data = get(bip(i), 'UserData');
    if ~isempty(data)
        if trainCategory(data) == 1
            set(bip(i), 'Color', 'r', 'Marker', 'o');
        else
            set(bip(i), 'Color', 'b', 'Marker', 'square');
        end
    end
end

end
