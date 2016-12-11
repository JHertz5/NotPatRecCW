%% Multiclass SVM

% Takes a long, long time (roughly 1:30 min)

% OVOSVM - one vs one SVM
% OVRSVM - one vs rest SVM

%% Clean up

clear all
close all
clc

if contains(pwd, 'NotPatRecCW')
    dataPath = strcat( extractBefore(pwd, 'NotPatRecCW'), 'NotPatRecCW/data');
    addpath(char(dataPath));
else
    fprintf('Move to NotPatRecCW directory\n');
end

%% Load Items

load Separated_Data.mat
load face.mat

%% Set test parameter
testImage = 31; % <- the only user variable here

testingImage1 = testing(:,(testImage-1)*2+1)';
testingImage2 = testing(:,(testImage-1)*2+2)';

%% Compute One vs One SVM with my function
[class1] = OVOSVM(testingImage1,training);
[class2] = OVOSVM(testingImage2,training);

if class1 == testImage
    fprintf('First image recognised correctly!\n');
else
    fprintf('First image not recognised...\n');
end

if class2 == testImage
    fprintf('Second image recognised correctly!\n');
else
    fprintf('Second image not recognised...\n');
end