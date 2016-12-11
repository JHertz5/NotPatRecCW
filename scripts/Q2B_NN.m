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
    fprintf('Move to NotPatRecCW directory\n');
end

numEigs = 150;

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
% The class of each image can be found by ceil(columnIndex / trainingClassSize)

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

classAssignment_ideal = ceil( 0.5 : 0.5 : ((size(trainingNorm,2))/trainingClassSize) );

minError = -ones(size(testingNorm,2),1, 'double');
classAssignment_real = zeros(1, size(testingNorm,2), 'double');

for testingFaceIndex = 1:size(testingNorm,2)
    for trainingFaceIndex = 1:size(trainingNorm,2)
        tempError = norm(w_testing(:,testingFaceIndex)-w_n(:,trainingFaceIndex));
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
    firstSuccessFace_matrix = zeros(faceH, faceW, 'double');
    firstSuccessExample_matrix = zeros(faceH, faceW, 'double');
    firstFailureFace_matrix = zeros(faceH, faceW, 'double');
    firstFailureExample_matrix = zeros(faceH, faceW, 'double');

    firstSuccessIndex = find(accuracyVector == true, 1);
    firstSuccessClassIndex = classAssignment_real(firstSuccessIndex);
    firstFailureIndex = find(accuracyVector == false, 1);
    firstFailureClassIndex = classAssignment_real(firstFailureIndex);

    for i = 1:faceW %extract image one line at a time
        lineStart = (i-1)*faceH + 1;
        lineEnd = i*faceH;
        
        firstSuccessFace_matrix(1:faceH,i) = rot90(testing((lineStart:lineEnd), firstSuccessIndex), 2);
        firstSuccessExample_matrix(1:faceH,i) = rot90(training((lineStart:lineEnd), firstSuccessClassIndex), 2);
        firstFailureFace_matrix(1:faceH,i) = rot90(testing((lineStart:lineEnd), firstFailureIndex), 2);
        firstFailureExample_matrix(1:faceH,i) = rot90(training((lineStart:lineEnd), firstFailureClassIndex), 2);
    end
    
    figure(1)
    

    
    subplot(2,2,1)
    h = pcolor(firstSuccessFace_matrix);
    set(h,'edgecolor','none');
    colormap gray
    shading interp
    ylabel('First Success Case')
    xlabel('Testing Face')
    set(gca,'XtickLabel',[],'YtickLabel',[]);
    
    subplot(2,2,2)
    h = pcolor(firstSuccessExample_matrix);
    set(h,'edgecolor','none');
    colormap gray
    shading interp
    xlabel('Assigned Class Example')
    set(gca,'XtickLabel',[],'YtickLabel',[]);
    
    subplot(2,2,3)
    h = pcolor(firstFailureFace_matrix);
    set(h,'edgecolor','none');
    colormap gray
    shading interp
    ylabel('First Failure Case')
    xlabel('Testing Face')
    set(gca,'XtickLabel',[],'YtickLabel',[]);
    
    subplot(2,2,4)
    h = pcolor(firstFailureExample_matrix);
    set(h,'edgecolor','none');
    colormap gray
    shading interp
    xlabel('Assigned Class Example')
    set(gca,'XtickLabel',[],'YtickLabel',[]);

end