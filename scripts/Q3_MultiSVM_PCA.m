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

% subtract mean face from testing faces
meanFace = mean(training,2);
testingNorm = testing - meanFace;

%% Project testing faces onto eigenspace

testingProjections = zeros(numEigs, size(testingNorm,2), 'double');
for n = 1:size(testingNorm,2)
        testingProjections(:,n) = (testingNorm(:,n)'*eigVecs_best(:,1:numEigs))';
end

%% Set test parameter
testImageIndex = 31; % <- the only user variable here

testingImage1 = testingProjections(:,(testImageIndex-1)*2+1)';
testingImage2 = testingProjections(:,(testImageIndex-1)*2+2)';

%% Compute One vs One SVM with my function
[class1] = OVOSVM(testingImage1,trainingProjections);
[class2] = OVOSVM(testingImage2,trainingProjections);

if class1 == testImageIndex
    fprintf('First image recognised correctly!\n');
else
    fprintf('First image not recognised...\n');
end

if class2 == testImageIndex
    fprintf('Second image recognised correctly!\n');
else
    fprintf('Second image not recognised...\n');
end