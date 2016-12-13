function [faceMatrix] = PlotFaceVector( faceWidth, faceHeight, faceVector )
%Function to plot a face vector

faceMatrix = zeros(faceHeight, faceWidth, 'double');

for i = 1:faceWidth %extract image one line at a time
    lineStart = (i-1)*faceHeight + 1;
    lineEnd = i*faceHeight;
    
    faceMatrix(1:faceHeight,i) = rot90(faceVector(lineStart:lineEnd), 2);
    
end

h = pcolor(faceMatrix);
set(h,'edgecolor','none');
colormap gray
shading interp
%ylabel('First Success Case')
set(gca,'XtickLabel',[],'YtickLabel',[]);


end
