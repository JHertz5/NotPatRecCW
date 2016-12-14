function [ ] = PlotConfusionMatrix( targetData, outputData )

numClasses = size(outputData, 1);
numTests = size(outputData, 2);

confusionMatrix = zeros(numClasses); % create matrix of zeros

for testIndex = 1:numTests
    targetClass = find(targetData(:,testIndex) == 1);
    outputClass = find(outputData(:,testIndex) == 1);
    
    confusionMatrix(outputClass, targetClass) = confusionMatrix(outputClass, targetClass) + 1;
end

% confusionMatrix is the true version, formatCM is a copy to be formatted for plotting
formatConfusionMatrix = -confusionMatrix; %set everything to negative

for i = 1:size(formatConfusionMatrix, 2)
    formatConfusionMatrix(i,i) = -formatConfusionMatrix(i,i) + 1; % set diagonal cells back to positive
end

pcolor(formatConfusionMatrix) %plot format copy

% set up colourmap
map = [ 1,      0,      0;
        0.5,    0,      0;
        0.2,    0,      0;
        0,      0.2,    0;
        0,      0.5,    0;
        0,      1,      0];

colormap(map)
labels = {'2 cases incorrect', '1 case incorrect', '0 cases incorrect', '0 cases correct', '1 case correct', '2 cases correct'};
lcolorbar(labels)
xlabel('Target Class')
ylabel('Output Class')
set(gca,'YDir','reverse')

end
