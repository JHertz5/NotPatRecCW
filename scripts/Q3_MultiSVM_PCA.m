%% Multiclass SVM

% Takes a long, long time (roughly 1:30 min)

% OVOSVM - one vs one SVM
% OVRSVM - one vs rest SVM

%% Clean up

close all
clc
%clear variables

if contains(pwd, 'NotPatRecCW')
    dataPath = strcat( extractBefore(pwd, 'NotPatRecCW'), 'NotPatRecCW/data'); 
    addpath(char(dataPath));
else
    fprintf('Move to NotPatRecCW directory\n');
end

%% Load Items

load Separated_Data.mat
load face.mat
Q1B_PCA

%% Normalise testing faces


faceW = 46; faceH = 56;
trainingClassSize = 8; %number of faces per class in training data
testingGroupSize = 2; %number of faces per class in testing data

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

classAssignment_ideal = ceil( 0.5 : 0.5 : ((size(trainingNorm,2))/trainingClassSize) );
classAssignment_real = zeros(1, 2*numClasses);

for testClassIndex = 1:numClasses
    
    testingImage1 = testingProjections(:, (testClassIndex-1)*2+1)';
    testingImage2 = testingProjections(:,  (testClassIndex-1)*2+2)';
    
    %% Compute One vs One SVM with my function
    [classAssignment1] = OVOSVM(testingImage1,testClassIndex,trainingProjections);
    [classAssignment2] = OVOSVM(testingImage2,testClassIndex,trainingProjections);
%     [classAssignment1] = OVASVM(testingImage1, testClassIndex, trainingProjections);
%     [classAssignment2] = OVASVM(testingImage2, testClassIndex, trainingProjections);
    
    %record results
    classAssignment_real(testClassIndex*2 -1) = classAssignment1;
    classAssignment_real(testClassIndex*2) = classAssignment2;

    if classAssignment1 == testClassIndex
        fprintf('Class %i - First image recognised correctly!\n', testClassIndex);
        accuracyVector(testClassIndex*2-1) = true;
    else
        fprintf('Class %i - First image not recognised: %i \n', testClassIndex, classAssignment1);
        accuracyVector(testClassIndex*2-1) = false;
    end
    
    if classAssignment2 == testClassIndex
        fprintf('Class %i - Second image recognised correctly!\n', testClassIndex);
        accuracyVector(testClassIndex*2) = true;
    else
        fprintf('Class %i - Second image not recognised: %i\n', testClassIndex, classAssignment2);
        accuracyVector(testClassIndex*2) = false;
    end
end

%% Plot Confusion Matrix

numTests = 2*numClasses;

confusion_groundTruth = zeros(numClasses, numTests);
confusion_resultsData = zeros(numClasses, numTests);

for i = 1:numTests
    confusion_groundTruth(classAssignment_ideal(i), i) = true;
    confusion_resultsData(classAssignment_real(i), i) = true;
end

figure
PlotConfusionMatrix(confusion_groundTruth, confusion_resultsData);
