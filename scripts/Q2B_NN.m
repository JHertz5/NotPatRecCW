%%
%Q2B using Nearest Neighbour

% clean up
clc
close all
%clear all

if contains(pwd, 'NotPatRecCW')
    dataPath = regexprep(pwd, 'NotPatRecCW', 'NotPatRecCW/data');
    addpath(char(dataPath));
else
    fprintf('Move to NotPatRecCW directory\n');
end

if ~exist('Separated_Data.mat', 'file')
    extract_ave_faces
end

Q1B_PCA

close all

load Separated_Data.mat
load Q1B_Eigen %using 1B because this method is more efficient

V = fliplr(V);

trainingClassSize = 8; %number of faces per class in training data
testingGroupSize = 2; %number of faces per class in training data
testingFaceIndex = 10;

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

%% Classify testing faces

classAssignment_ideal = ceil( 0.5 : 0.5 : ((size(trainingNorm,2))/trainingClassSize) );

minError = -ones(size(testingNorm,2),1, 'double');
classAssignment_real = zeros(1, size(testingNorm,2), 'double');

for testingFaceIndex = 1:size(testingNorm,2)
    for trainingFaceIndex = 1:size(trainingNorm,2)
        tempError = norm(testingProjections(:,testingFaceIndex)-trainingProjections(:,trainingFaceIndex));
        if tempError < minError(testingFaceIndex) || minError(testingFaceIndex) < 0
            minError(testingFaceIndex) = tempError;
            classAssignment_real(testingFaceIndex) = ceil(trainingFaceIndex/trainingClassSize);
        end
    end
    fprintf('Testing image %i is assigned class %i\n', testingFaceIndex, classAssignment_real(testingFaceIndex));
end


%% Compute accuracy

accuracyVector = (classAssignment_ideal == classAssignment_real);
successPercentage = 100 * sum(accuracyVector, 2) / size(testingNorm, 2)

%% Plot example success and example failure

if (exist('showPlots', 'var') && showPlots == true)

    faceW = 46; faceH = 56;
    
    firstSuccessIndex = find(accuracyVector == true, 1);
    firstSuccessClassIndex = classAssignment_real(firstSuccessIndex);
    firstFailureIndex = find(accuracyVector == false, 1);
    firstFailureClassIndex = classAssignment_real(firstFailureIndex);
    
    figure(1)
    
    subplot(1,4,1)
    PlotFaceVector(faceW, faceH, testing(:, firstSuccessIndex));
    
    subplot(1,4,2)
    PlotFaceVector(faceW, faceH, training(:, firstSuccessClassIndex));
    
    subplot(1,4,3)
    PlotFaceVector(faceW, faceH, testing(:, firstFailureIndex));
    
    subplot(1,4,4)
    PlotFaceVector(faceW, faceH, training(:, firstFailureClassIndex));
    
end

%% Plot confusion matrix

numClasses = size(training, 2)/trainingClassSize;
numTests = size(accuracyVector, 2);

confusion_groundTruth = zeros(numClasses, numTests);
confusion_resultsData = zeros(numClasses, numTests);

for i = 1:numTests
    confusion_groundTruth(classAssignment_ideal(i), i) = true;
    confusion_resultsData(classAssignment_real(i), i) = true;
end

%figure(2);
%plotconfusion(confusion_groundTruth, confusion_resultsData)