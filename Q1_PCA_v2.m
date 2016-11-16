%%
% Second Attempt at Q1 PCA

% clean up
clc
close all
clear all

% load partitioned data
load Separated_Data.mat

% change plots to 1 to get images
plots = 1;

%% Normalise and plot mean face

% subtract mean face from training faces
mean_Face = mean(training,2);
training_t = training - mean_Face;

% plot mean face
if plots == 1
    figure(1)
    for i = 1:46 %extract image one line at a time
        mean_Face_m(1:56,i) = rot90(mean_Face((i-1)*56+1:i*56), 2);
    end
    
    h = pcolor(mean_Face_m)
    set(h,'edgecolor','none');
    colormap gray
    title('Average Face','fontsize',20)
end

%% Do math and all

% Calculate Covariance Matrix
[len wid] = size(training_t);
faceCov = (training_t*training_t')/wid;

% Find eigenvalues and eigenvectors, D is a diagonal matrix - pointless
[V,D] = eig(faceCov);

% Move the diagonal onto an array
for i = 1:length(D)
    eigVals(i) = D(i,i);
end

%% plot eig vals

% Plot all eigenvalues sorted. Number of non-zero eigenvalues should be N -
% 1, where N is number of training data (416 in this case)
if plots == 1
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
num_eigs = 50;
[sortedEigs,sortedIdx] = sort(eigVals,'descend');
bestIdx = sortedIdx(1:num_eigs);
eigVals_best = sortedEigs(1:num_eigs);
eigVecs_best = V(:,bestIdx);

%% plot 10 eigenfaces

if plots == 1
    figure(3)
    for j = 1:10
        for i = 1:46 %extract image one line at a time
            eigFace(1:56,i,j) = rot90(eigVecs_best((i-1)*56+1:i*56,j), 2);
            
        end
        subplot(2,5,j)
        h = pcolor(eigFace(:,:,j));
        set(h,'edgecolor','none');
        colormap gray
    end
end