%%
%Q2A using Q1A

% clean up
clc
close all
%clear all

if contains(pwd, 'NotPatRecCW')
    dataPath = strcat( extractBefore(pwd, 'NotPatRecCW'), 'NotPatRecCW/data');
    addpath(char(dataPath));
else
    printf('Move to NotPatRecCW directory\n');
end

load Separated_Data.mat
load Q1A_Eigen
V = fliplr(V);

%% Reconstruction

trainFaceIdx = 1; %index of a face from training set to be reconstructed

reconstructedFace = meanFace;
for n = 1:numEigs
     reconstructedFace = reconstructedFace + trainingProjections(n,trainFaceIdx)*eigVecs_best(:,n);
end

%% Plot for comparison

faceW = 46; faceH = 56;

if (exist('showPlots', 'var') && showPlots == true)
    figure(1)
    
    subplot(1,2,1)
    PlotFaceVector(faceW, faceH, training(:,trainFaceIdx));
    title('Original Face','fontsize',20)
    
    subplot(1,2,2)
    PlotFaceVector(faceW, faceH, reconstructedFace(:));
    title(['Reconstructed Face with ' num2str(numEigs) ' eigenfaces'],'fontsize',20)
    
else
    fprintf('No plots because showPlots != true\n')
end
ReconstructionError = norm(training(:,trainFaceIdx)-reconstructedFace)

%% Enter testing face for reconstruction

testingIdx = 1;
testFace = testing(:,testingIdx);

% subtract the mean
testFace = testFace - meanFace;

% project it onto eigenfaces
w_test = (testFace'*V(:,1:numEigs))';

% compare each wn with w_test to find min error -> resulting in
% indentification