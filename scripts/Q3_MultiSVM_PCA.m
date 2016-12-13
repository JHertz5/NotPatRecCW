%% Multiclass SVM

% Takes a long, long time (roughly 1:30 min)

% OVOSVM - one vs one SVM
% OVRSVM - one vs rest SVM

%% Clean up

close all
clc
%clear variables

if contains(pwd, 'NotPatRecCW')
    dataPath = regexprep(pwd, 'NotPatRecCW', 'NotPatRecCW/data');
    addpath(char(dataPath));
else
    fprintf('Move to NotPatRecCW directory\n');
end

%% Load Items

load Separated_Data.mat
load face.mat
Q1B_PCA

%% Normalise testing faces
tic;

faceW = 46; faceH = 56;

% subtract mean face from testing faces
meanFace = mean(training,2);
testingNorm = testing - meanFace;

%% Project testing faces onto eigenspace

testingProjections = zeros(numEigs, size(testingNorm,2), 'double');
for n = 1:size(testingNorm,2)
    testingProjections(:,n) = (testingNorm(:,n)'*eigVecs_best(:,1:numEigs))';
end

%% Set test loop

numClasses = size(testing, 2)/2;
accuracyVector = zeros(1, size(testing, 2), 'logical');

for testClassIndex = 1:1
    
    testingImage1 = testingProjections(:, (testClassIndex-1)*2+1)';
    testingImage2 = testingProjections(:,  (testClassIndex-1)*2+2)';
    
    %% Compute One vs One SVM with my function
    [classAssignment1] = OVOSVM(testingImage1,trainingProjections);
    [classAssignment2] = OVOSVM(testingImage2,trainingProjections);
    %[classAssignment1] = OVASVM(testingImage1, testClassIndex, trainingProjections);
    %[classAssignment2] = OVASVM(testingImage2, testClassIndex, trainingProjections);
    
    if classAssignment1 == testClassIndex
        fprintf('Class %i - First image recognised correctly!\n', testClassIndex);
        accuracyVector(testClassIndex*2) = true;
    else
        fprintf('Class %i - First image not recognised\n', testClassIndex);
        accuracyVector(testClassIndex*2) = false;
    end
    
    if classAssignment2 == testClassIndex
        fprintf('Class %i - Second image recognised correctly!\n', testClassIndex);
        accuracyVector(testClassIndex*2 + 1) = true;
    else
        fprintf('Class %i - image not recognised\n', testClassIndex);
        accuracyVector(testClassIndex*2 + 1) = false;
    end
end