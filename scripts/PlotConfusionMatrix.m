function [ ] = PlotConfusionMatrix( targetData, outputData )

numClasses = size(outputData, 1);
numTests = size(outputData, 2);

confusionMatrix = zeros(numClasses); % create matrix of zeros

% y axis has to be output class
% x axis has to be target class

for testIndex = 1:numTests
    targetClass = find(targetData(:,testIndex) == 1);
    outputClass = find(outputData(:,testIndex) == 1);
    confusionMatrix(outputClass, targetClass) = confusionMatrix(targetClass, outputClass) + 1;
end

% confusionMatrix is the true version, formatCM is a copy to be formatted for plotting
formatConfusionMatrix = -(confusionMatrix + ones(numClasses)); %set everything to negative

for i = 1:size(formatConfusionMatrix, 2)
    formatConfusionMatrix(i,i) = -formatConfusionMatrix(i,i); % set diagonal cells back to positive
end

pcolor(formatConfusionMatrix) %plot format copy

map = [ 1,      0,      0;
        0.5,    0,      0;
        0.2,    0,      0;
        0,      0,      0;
        0,      0,      0;
        0,      0.2,    0;
        0,      0.5,    0;
        0,      1,      0];

colormap(map)

xlabel('Target Class');
ylabel('Output Class');
set(gca,'YDir','reverse');


end

