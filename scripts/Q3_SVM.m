%% Support Vector Machines

clear all
close all
clc

if contains(pwd, 'NotPatRecCW')
    dataPath = strcat( extractBefore(pwd, 'NotPatRecCW'), 'NotPatRecCW/data');
    addpath(char(dataPath));
else
    printf('Move to NotPatRecCW directory\n');
end

load Separated_Data.mat
load face.mat
value = 1;
counter = 1;
for i = 1:416
    class(i) = value;
    counter = counter + 1;
    if counter == 9
        counter = 1;
        value = value + 1;
    end
end

trainV2 = [class; training];

%% extract just two classes

twoClass = trainV2(:,1:16);

% STEP1: Go to APPS, click om Classification Learner
% STEP2: Press New Session - > from Workspace
% STEP3: Click twoClass on the LHS and below it select Use Rows as
% Variables
% STEP4: In the window's Step two, set the first row "Improt as" to Response
% STEP5: Press Start Session
% STEP6: On the top bar, select any SVM you wat to use and press Train

%% Alternatively use this guy

% Select which classes to use <- The only user bit here. Rest you dont need
% to modify (other than optimise)
class2 = 10;

% set up trackers of guesses
correct = 0;
incorrect = 0;

% put the classes of interest on the matrix transpose -> rows contain faces
% now
binaryTrain = [training(:,(class1-1)*8+1:(class1-1)*8+8) training(:,(class2-1)*8+1:(class2-1)*8+8)]';
binaryTest = [testing(:,(class1-1)*2+1:(class1-1)*2+2) training(:,(class2-1)*2+1:(class2-1)*2+2)]';
trainFlags = [class1*ones(1,8) class2*ones(1,8)];
testFlags = [class1*ones(1,2) class2*ones(1,2)];

% Compute the SVM model
SVMModel = fitcsvm(binaryTrain,trainFlags,'KernelFunction','linear','Standardize',true);

% Test its correctness on the training data
for i = 1:16
    [label,~] = predict(SVMModel,binaryTrain(i,:));
    if label == trainFlags(i)
        correct = correct + 1;
    else
        incorrect = incorrect + 1;
    end
end
TrainCorrectness = correct*100/16

correct = 0;
incorrect = 0;

% Test it on the testing data
for i = 1:4
    [label,~] = predict(SVMModel,binaryTest(i,:));
    if label == testFlags(i)
        correct = correct + 1;
    else
        incorrect = incorrect + 1;
    end
end

TestCorrectness = correct*100/4