%% Q1 - PCA for each set of faces
% Dont think this is right. Also we dont have to do it for all 52 classes.
% Just use 2 or 3 different faces instead of 52...

close all
clear all
clc

load Separated_Data

detrend_faces = [];
mean_faces = zeros(2576,52,'double');
faceCov = zeros(2576,2576,52,'double');

for i = 1:52
    % Compute mean face and subtract it from all 8 faces. Repeat for 52
    % different faces
    mean_faces(:,i) = mean(training(:,(i-1)*8+1:i*8),2);
    detrend_faces = [detrend_faces (training(:,(i-1)*8+1:i*8) - mean_faces(:,i))];
    
    % Compute covariance matrix for each face set
    % This thing weighs 20.5 GB (lol)
    faceCov(:,:,i) = 0.125*(detrend_faces(:,(i-1)*8+1:i*8)*detrend_faces(:,(i-1)*8+1:i*8)');
    
    [V(:,:,i),D(:,:,i)] = eig(faceCov(:,:,i));
end

%% Move the eigenvalues

for i = 1:2576
    for j = 1:52
        EigVals(i,j) = D(i,i,j);
    end
end

%% keep best eigenvalues and corresponding eigenvectors (7 in this case)

BestEigVals = EigVals(2570:end,:);
BestEigVecs = V(:,2570:2576,:);
figure(1)
for j = 1:7
    for i = 1:46
        indvFaces2(1:56,i,j) = rot90(BestEigVecs((i-1)*56+1:i*56, j, 1),2);
    end
    subplot(2,4,j)
    h = pcolor(indvFaces2(:,:,j));
    set(h,'edgecolor','none');
    colormap gray
end