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

if ~(exist('numEigs', 'var'))
    numEigs = 416;
end

%% Normalise and plot mean face

faceW = 46; faceH = 56;

% subtract mean face from training faces
meanFace = mean(training,2);
trainingNorm = training - meanFace;

meanFace_matrix = zeros(faceH, faceW, 'double');

% plot mean face
if (exist('showPlots', 'var') && showPlots == true)
    figure(1)
    for i = 1:faceW %extract image one line at a time
        lineStart = (i-1)*faceH + 1;
        lineEnd = i*faceH;
        meanFace_matrix(1:faceH,i) = rot90(meanFace(lineStart:lineEnd), 2);
    end
    
    h = pcolor(meanFace_matrix);
    set(h,'edgecolor','none');
    colormap gray
    shading interp
    title('Average Face','fontsize',20)
end

%% Do math and all

% Calculate Covariance Matrix
tic;
N = size(trainingNorm, 2);
faceCov = (trainingNorm'*trainingNorm)/N;
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

[sortedEigs,sortedIdxList] = sort(eigVals,'descend');
bestIdxList = sortedIdxList(1:numEigs);
eigVals_best = sortedEigs(1:numEigs); % extract top M eigenvalues
eigVecsB_best = V(:,bestIdxList); % extract best M eigenvectors

%% use A'A eigenvectors to calculate AA' eigenvectors
% A'A and AA' have the same eigenvalues

eigVecs_best = trainingNorm*eigVecsB_best;
t2 = toc
%normalise face vectors
for i=1:numEigs
   eigVecs_best(:,i) = eigVecs_best(:,i) /sqrt(eigVals_best(i));
end
t_done2 = toc()
%% plot 10 eigenfaces

eigFace = zeros(faceH, faceW, 3, 'double');
if (exist('showPlots', 'var') && showPlots == true)
    figure(3)
    for j = 1:3
        for i = 1:faceW %extract image one line at a time
            lineStart = (i-1)* faceH + 1;
            lineEnd = i*faceH;
            eigFace(1:faceH,i,j) = rot90(eigVecs_best(lineStart:lineEnd,j), 2);
        end
        subplot(1,3,j)
        h = pcolor(eigFace(:,:,j));
        set(h,'edgecolor','none');
        colormap gray
        shading interp
        set(findobj(gcf, 'type','axes'), 'Visible','off')
    end
else
    fprintf('No plots because showPlots != true\n')
end

if (exist('dataPath', 'var'))
    save(char(strcat(dataPath, '/Q1B_Eigen')),'eigVals_best','eigVecs_best','V','trainingNorm','meanFace')
else
    save('Q1B_Eigen','eigVals_best','eigVecs_best','V','trainingNorm','meanFace')
end
