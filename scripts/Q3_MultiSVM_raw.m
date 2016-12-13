%% Multiclass SVM

% Takes a long, long time (roughly 1:30 min)

% OVOSVM - one vs one SVM
% OVRSVM - one vs rest SVM

%% Clean up

clear all
close all
clc

if contains(pwd, 'NotPatRecCW')
    dataPath = regexprep(pwd, 'NotPatRecCW', 'NotPatRecCW/data');
    addpath(char(dataPath));
else
    fprintf('Move to NotPatRecCW directory\n');
end

%% Load Items

load Separated_Data.mat
load face.mat

%% Set test loop

numClasses = size(testing, 2)/2;
accuracyVector = zeros(1, size(testing, 2), 'logical');

for testClassIndex = 1:numClasses
    
    testingImage1 = testing(:, (testClassIndex-1)*2+1)';
    testingImage2 = testing(:,  (testClassIndex-1)*2+2)';
    
    %% Compute One vs One SVM with my function
    [classAssignment1] = OVOSVM(testingImage1,testClassIndex,training);
    [classAssignment2] = OVOSVM(testingImage2,testClassIndex,training);
    %[classAssignment1] = OVASVM(testingImage1, testClassIndex, training);
    %[classAssignment2] = OVASVM(testingImage2, testClassIndex, training);
    
    if classAssignment1 == testClassIndex
        fprintf('Class %i - First image recognised correctly!\n', testClassIndex);
        accuracyVector(testClassIndex*2) = true;
    else
        fprintf('Class %i - First image not recognised: %i \n', testClassIndex, classAssignment1);
        accuracyVector(testClassIndex*2) = false;
    end
    
    if classAssignment2 == testClassIndex
        fprintf('Class %i - Second image recognised correctly!\n', testClassIndex);
        accuracyVector(testClassIndex*2 + 1) = true;
    else
        fprintf('Class %i - Second image not recognised: %i\n', testClassIndex, classAssignment2);
        accuracyVector(testClassIndex*2 + 1) = false;
    end
end
