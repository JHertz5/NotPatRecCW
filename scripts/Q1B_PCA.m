%%

% clean up
clc
close all
clear variables

% load partitioned data
load Separated_Data.mat

% change showPlots to true to get images
showPlots = true;

%% Normalise and plot mean face

faceW = 46; faceH = 56;

% subtract mean face from training faces
mean_Face = mean(training,2);
training_t = training - mean_Face;

mean_Face_m = zeros(faceH, faceW, 'double');

% plot mean face
if showPlots == true
    figure(1)
    for i = 1:faceW %extract image one line at a time
        lineStart = (i-1)*faceH + 1;
        lineEnd = i*faceH;
        mean_Face_m(1:faceH,i) = rot90(mean_Face(lineStart:lineEnd), 2);
    end
    
    h = pcolor(mean_Face_m);
    set(h,'edgecolor','none');
    colormap gray
    shading interp
    title('Average Face','fontsize',20)
end

%% Do math and all

% Calculate Covariance Matrix
N = size(training_t, 2);
faceCov = (training_t'*training_t)/N;

% Find eigenvalues and eigenvectors, D is a diagonal matrix - pointless
[V,D] = eig(faceCov);

eigVals = zeros(1, length(D));
% Move the diagonal onto an array
for i = 1:length(D)
    eigVals(i) = D(i,i);
end

%% plot eig vals

% Plot all eigenvalues sorted. Number of non-zero eigenvalues should be N -
% 1, where N is number of training data (416 in this case)
if showPlots == true
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
M = 50;
[sortedEigs,sortedIdx] = sort(eigVals,'descend');
bestIdx = sortedIdx(1:M);
eigVals_best = sortedEigs(1:M); % extract top M eigenvalues
eigVecs_best = V(:,bestIdx); % extract best M eigenvectors

%% use A'A eigenvectors to calculate AA' eigenvectors
% A'A and A'A have the same eigenvalues

eigFaceVecs = training_t*eigVecs_best;

%normalise face vectors
for i=1:M
   eigFaceVecs(:,i) = eigFaceVecs(:,i) / abs(norm(eigFaceVecs(:,i)));
end

%% plot 10 eigenfaces

eigFace = zeros(faceH, faceW, 10, 'double');
if showPlots == true
    figure(3)
    for j = 1:10
        for i = 1:faceW %extract image one line at a time
            lineStart = (i-1)* faceH + 1;
            lineEnd = i*faceH;
            eigFace(1:faceH,i,j) = rot90(eigFaceVecs(lineStart:lineEnd,j), 2);
        end
        subplot(2,5,j)
        h = pcolor(eigFace(:,:,j));
        set(h,'edgecolor','none');
        colormap gray
        shading interp
        set(findobj(gcf, 'type','axes'), 'Visible','off')
    end
end