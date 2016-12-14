function [assignedClass] = OVASVM(testingImage, testLabel, training)

scores = zeros(52,1);

for i = 1:52
    class1 = i;
    
    trainFlags = -ones(size(training,2),1);
    trainFlags( (class1-1)*8+1:(class1-1)*8+8 ) = 1;
    
    SVMModel = svmtrain(trainFlags, training', '-t 0 -q');

    [~,~,scores(i)] = svmpredict(testLabel,testingImage,SVMModel,'-q');

end

[~, assignedClass] = max(scores);
end