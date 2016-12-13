%%
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

% load partitioned data
load Separated_Data.mat

%% Normalise and plot mean face

faceW = 46; faceH = 56;

% subtract mean face from training faces
meanFace = mean(training,2);
trainingNorm = training - meanFace;

meanFace_matrix = zeros(faceH, faceW, 'double');

% plot mean face
if (exist('showPlots', 'var') && showPlots == true)
    figure(1)
    PlotFaceVector(faceW, faceH, meanFace(:));
    title('Average Face','fontsize',20)
end

%% Do math and all

% Calculate Covariance Matrix
N = size(trainingNorm, 2);
faceCov = (trainingNorm'*trainingNorm);
% Find eigenvalues and eigenvectors, D is a diagonal matrix - pointless
[V,D] = eig(faceCov);

eigVals = diag(D); % move D into an array

%% plot eig vals

% Plot all eigenvalues sorted. Number of non-zero eigenvalues should be N -
% 1, where N is number of training data (416 in this case)
if (exist('showPlots', 'var') && showPlots == true)
    figure(2)
    plot(sort(eigVals,'descend'),'linewidth',2)
    set(gca,'YScale','log')
    title('Eigenvalues sorted','fontsize',20)
    xlabel('Index','fontsize',20)
    ylabel('Eigenvalue','fontsize',20)
    xlim([0 415])
    grid on
    grid minor
end

%% get M best eigenvectors/values

% technically the eigenvalues are presorted in the ascending order. But
% just to be sure sort them again
numEigs = 150;
[sortedEigs,sortedIdxList] = sort(eigVals,'descend');
bestIdxList = sortedIdxList(1:numEigs);
eigVals_best = sortedEigs(1:numEigs); % extract top M eigenvalues
eigVecsB_best = V(:,bestIdxList); % extract best M eigenvectors

%% use A'A eigenvectors to calculate AA' eigenvectors
% A'A and AA' have the same eigenvalues

eigVecs_best = trainingNorm*eigVecsB_best;
%normalise face vectors
for i=1:numEigs
   eigVecs_best(:,i) = eigVecs_best(:,i) /sqrt(eigVals_best(i));
end

%% Find trainingProjections for each normalised training face

trainingProjections = zeros(numEigs, size(trainingNorm,2), 'double');
for n = 1:size(trainingNorm,2)
        trainingProjections(:,n) = (trainingNorm(:,n)'*eigVecs_best(:,1:numEigs))';
end

% trainingProjections has now dimensions numEigs by size(trainigNorm,2) -> decresed
% dimensionality to save on space, memory, computation time but to preserve
% maximum feature variance

% Columns of trainingProjections represent different face images. 
% Face images are classed in groups of 8
% The class of each image can be found by ceil(columnIndex / trainingClassSize)

%% plot 10 eigenfaces

eigFace = zeros(faceH, faceW, 3, 'double');
if (exist('showPlots', 'var') && showPlots == true)
    figure(3)
    for j = 1:3
        subplot(1,3,j)
        PlotFaceVector(faceW, faceH, eigVecs_best(:, j));
    end
else
    %fprintf('No plots because showPlots != true\n')
end

%% Save data

if (exist('dataPath', 'var'))
    save(char(strcat(dataPath, '/Q1B_Eigen')),'eigVals_best','eigVecs_best','V','trainingNorm','meanFace','trainingProjections','numEigs')
else
    save('Q1B_Eigen','eigVals_best','eigVecs_best','V','trainingNorm','meanFace','trainingProjections','numEigs')
end
