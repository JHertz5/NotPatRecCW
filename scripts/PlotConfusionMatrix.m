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
map = [%r,      g,      b;
        1,      0,      0;
        0.5,    0,      0;
        0.2,    0,      0;
        0,      0.2,    0;
        0,      0.5,    0;
        0,      1,      0];

set(0,'defaulttextinterpreter','latex')
set(gca,'FontSize', 10)
colormap(map)
labels = {'2 cases', '1 case', '0 cases', '0 cases', '1 case', '2 cases'};
lcolorbar(labels, 'FontSize', 20)
xlabel('Target Class', 'FontSize', 25)
ylabel('Output Class', 'FontSize', 25)
set(gca,'YDir','reverse')

end
