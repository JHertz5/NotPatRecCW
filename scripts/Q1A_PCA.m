%%
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
N = size(trainingNorm, 2);
faceCov = (trainingNorm*trainingNorm')/N;

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
    xlim([0 415])
    grid on
    grid minor
end

%% get M best eigenvectors/values

% technically the eigenvalues are presorted in the ascending order. But
% just to be sure sort them again
M = 200;
[sortedEigs,sortedIdxList] = sort(eigVals,'descend');
bestIdxList = sortedIdxList(1:M);
eigVals_best = sortedEigs(1:M); % extract top M eigenvalues
eigVecs_best = V(:,bestIdxList); % extract best M eigenvectors

%% plot 10 eigenfaces

eigFace = zeros(faceH, faceW, 10, 'double');
if (exist('showPlots', 'var') && showPlots == true)
    figure(3)
    for j = 1:10
        for i = 1:faceW %extract image one line at a time
            lineStart = (i-1)* faceH + 1;
            lineEnd = i*faceH;
            eigFace(1:faceH,i,j) = rot90(eigVecs_best(lineStart:lineEnd,j), 2);
        end
        subplot(2,5,j)
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
    save(char(strcat(dataPath, '/Q1A_Eigen')),'eigVals_best','eigVecs_best','V','trainingNorm','meanFace')
else
    save('Q1A_Eigen','eigVals_best','eigVecs_best','V','trainingNorm','meanFace')
end