%% Support Vector Machines
% Currently only for raw intensity vectors
% TO BE DONE: Confusion Matrices, MultiClass SVA - one v. one & one v. all,

clear all
close all
clc

if contains(pwd, 'NotPatRecCW')
    dataPath = regexprep(pwd, 'NotPatRecCW', 'NotPatRecCW/data');
    addpath(char(dataPath));
else
    fprintf('Move to NotPatRecCW directory\n');
end

load Separated_Data.mat
load face.mat
load Q1B_Eigen.mat

%% Project testing faces onto eigenspace
testingNorm = testing - meanFace;
testingProjections = zeros(numEigs, size(testingNorm,2), 'double');
for n = 1:size(testingNorm,2)
        testingProjections(:,n) = (testingNorm(:,n)'*eigVecs_best(:,1:numEigs))';
end

%% Set up training data

class = ones(1,416);
trainingClassSize = 8; %number of faces per class in training data
testingClassSize = 2;
numClasses = size(trainingProjections, 2)/trainingClassSize;

for i = 1:numClasses
    lineStart = ((i-1)*trainingClassSize) + 1;
    lineEnd = i*trainingClassSize;
    class(1, lineStart:lineEnd) = i*ones(1,trainingClassSize);
end
trainV2 = [class; trainingProjections];


%% extract just two classes

twoClass = trainV2(:,1:16);

% STEP1: Go to APPS, click om Classification Learner
% STEP2: Press New Session - > from Workspace
% STEP3: Click twoClass on the LHS and below it select Use Rows as
% Variables (make sure to run the script first)
% STEP4: In the window's Step Two, set the first row "Import as" to Response
% STEP5: Press Start Session
% STEP6: On the top bar, select any SVM you want to use and press Train

%% Alternatively use this guy

% Select which classes to use <- The only user bit here. Rest you dont need
% to modify (other than optimise)
class1 = 32;
class2 = 10;

% set up trackers of guesses
correct = 0;
incorrect = 0;

% put the classes of interest on the matrix transpose -> rows contain faces
% now
binaryTrain = [trainingProjections(:,(class1-1)*8+1:(class1-1)*8+8) trainingProjections(:,(class2-1)*8+1:(class2-1)*8+8)]';
binaryTest = [testingProjections(:,(class1-1)*2+1:(class1-1)*2+2) testingProjections(:,(class2-1)*2+1:(class2-1)*2+2)]';
trainFlags = [class1*ones(1,8) class2*ones(1,8)];
confTrain = [zeros(1,8) ones(1,8)];
testFlags = [class1*ones(1,2) class2*ones(1,2)];
confTest = [zeros(1,2) ones(1,2)];

conf1 = zeros(1, 16);
conf2 = zeros(1, 4);

% Compute the SVM model
SVMModel = fitcsvm(binaryTrain,trainFlags,'KernelFunction','linear','Standardize',true);

% Test its correctness on the training data
for i = 1:16
    [label1,~] = predict(SVMModel,binaryTrain(i,:));
    if label1 == trainFlags(i)
        correct = correct + 1;
    else
        incorrect = incorrect + 1;
    end
    
    if label1 == class1
        conf1(i) = 0;
    else
        conf1(i) = 1;
    end
end

TrainCorrectness = correct*100/16;
figure(1)
plotconfusion(confTrain,conf1,'Confusion Matrix of Training Data')

correct = 0;
incorrect = 0;

% Test it on the testing data
for i = 1:4
    [label2,~] = predict(SVMModel,binaryTest(i,:));
    if label2 == testFlags(i)
        correct = correct + 1;
    else
        incorrect = incorrect + 1;
    end
    
    if label2 == class1
        conf2(i) = 0;
    else
        conf2(i) = 1;
    end
end
figure(2)
plotconfusion(confTest,conf2,'Confusion Matrix of Testing Data')

TestCorrectness = correct*100/4;