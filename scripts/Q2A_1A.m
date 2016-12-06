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

%numEigs = 20;
%numEigs = 50;
numEigs = 80;

%% Calculate wn = [an1 an2 ... anM]', ani = normFace_n'*ui

w = zeros(numEigs, 416, 'double');
for n = 1:size(trainingNorm,2)
        w(:,n) = (trainingNorm(:,n)'*V(:,1:numEigs))';
end
% wn has now dimensions numEigs by size(trainigNorm,2) -> decresed
% dimensionality to save on space, memory, computation time but to preserve
% maximum feature variance

%%

% Reconstruction

trainFaceIdx = 1; %index of a face from training set to be reconstructed

reconstructedFace = zeros(1,2576);

for n = 1:numEigs
     reconstructedFace = reconstructedFace + w(n,trainFaceIdx)*V(:,n);
end
reconstructedFace = reconstructedFace + meanFace;

%% Plot for comparison

faceW = 46; faceH = 56;
origFace = zeros(faceH, faceW, 'double');
recoFace = zeros(faceH, faceW, 'double');

if (exist('showPlots', 'var') && showPlots == true)
    figure(1)
    for i = 1:faceW %extract image one line at a time
        lineStart = (i-1)*faceH + 1;
        lineEnd = i*faceH;
        origFace(1:faceH,i) = rot90(training(lineStart:lineEnd,trainFaceIdx), 2);
        recoFace(1:faceH,i) = rot90(reconstructedFace(lineStart:lineEnd), 2);
    end
    
%     subplot(1,4,1)
%     h = pcolor(origFace);
%     set(h,'edgecolor','none');
%     colormap gray
%     shading interp
%     title('Original Face','fontsize',20)
    
    subplot(1,4,4)
    h = pcolor(recoFace);
    set(h,'edgecolor','none');
    colormap gray
    shading interp
    title(['Reconstructed Face with ' num2str(numEigs) ' eigenfaces'],'fontsize',20)
    set(findobj(gcf, 'type','axes'), 'Visible','off')
else
    fprintf('No plots because showPlots != true\n')
end
% ReconstructionError = norm(training(:,trainFaceIdx)-reconstructedFace)

%% Enter testing face for reconstruction

testingIdx = 1;
testFace = testing(:,testingIdx);

% subtract the mean
testFace = testFace - meanFace;

% project it onto eigenfaces
w_test = (testFace'*V(:,1:numEigs))';

% compare each wn with w_test to find min error -> resulting in
% indentification