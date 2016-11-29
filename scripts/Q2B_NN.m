%%
%Q2B using Nearest Neighbour

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

if ~exist('Separated_Data.mat', 'file')
    extract_ave_faces
end
if ~exist('Q1B_Eigen.mat', 'file')
    Q1B_PCA
end

load Separated_Data.mat
load Q1B_Eigen %using 1B because this method is more efficient

V = fliplr(V);

numEigs = 200;

%% Calculate wn = [an1 an2 ... anM]', ani = normFace_n'*ui
% representing projection on the eigenspace

w_n = zeros(numEigs, size(trainingNorm,2), 'double');
for n = 1:size(trainingNorm,2)
        w_n(:,n) = (trainingNorm(:,n)'*eigVecs_best(:,1:numEigs))';
end
% w_n has now dimensions numEigs by size(trainigNorm,2) -> decresed
% dimensionality to save on space, memory, computation time but to preserve
% maximum feature variance

% Columns of w_n represent different face images. 
% Face images are classed in groups of 8
% The class of each image can be found by ceil(columnIndex / classSize)

%% Plot testing face

testingFaceIndex = 1;

faceW = 46; faceH = 56;
face_matrix = zeros(faceH, faceW, 'double');

if (exist('showPlots', 'var') && showPlots == true)
    figure(1)
    for i = 1:faceW %extract image one line at a time
        lineStart = (i-1)*faceH + 1;
        lineEnd = i*faceH;
        face_matrix(1:faceH,i) = rot90(testing((lineStart:lineEnd),testingFaceIndex), 2);
    end
    
    h = pcolor(face_matrix);
    set(h,'edgecolor','none');
    colormap gray
    shading interp
    title('Testing Face')
end


%% Normalise testing faces

faceW = 46; faceH = 56;

% subtract mean face from testing faces
meanFace = mean(training,2);
testingNorm = testing - meanFace;

%% Project testing faces onto eigenspace

w_testing = zeros(numEigs, size(testingNorm,2), 'double');
for n = 1:size(testingNorm,2)
        w_testing(:,n) = (testingNorm(:,n)'*eigVecs_best(:,1:numEigs))';
end

%% Classify testing faces


minError = -ones(size(testingNorm,2),1, 'double');
classAssignment = zeros(size(testingNorm,2), 1, 'double');

for trainingFaceIndex = 1:size(trainingNorm,2)
    tempError = norm(w_testing(:,testingFaceIndex)-w_n(:,trainingFaceIndex));
    if tempError < minError(testingFaceIndex) || minError(testingFaceIndex) < 0
        minError(testingFaceIndex) = tempError;
        classAssignment(testingFaceIndex) = ceil(trainingFaceIndex/8);
    end
end