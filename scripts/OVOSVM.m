function [assignedClass] = OVOSVM(testingImage, testLabel, training)

% One vs One SVM requires us to carry out (for classSize = 52)
% 51+50+49...+1 trainings -> that is T = (classSize + 1)*classSize/2 trainings.
% Then a testing face is fed into each of T models and votes are counted.
% Each model will return a value {class1, class2} -> number of occurences will be counted
% The one with biggest amount of votes will be the winner

votes = zeros(1,52);

for i = 1:51
    for j = i+1:52
        
        % train each set of classes once -> 1 v 2 == 2 v 1
        class1 = i;
        class2 = j;
        
        % extract the right data for easier handling
        binaryTrain = [training(:,(class1-1)*8+1:(class1-1)*8+8) training(:,(class2-1)*8+1:(class2-1)*8+8)]';
        trainFlags = [class1*ones(1,8) class2*ones(1,8)]';
        
        % estimate model for those two classes
        SVMModel = svmtrain(trainFlags,binaryTrain ,'-t 2 -q');
        %SVMModel = fitcsvm(binaryTrain,trainFlags,'KernelFunction','linear','Standardize',true);
        
        [label,~,~] = svmpredict(testLabel,testingImage,SVMModel, '-q');
        %[label,~] = predict(SVMModel,testingImage);
        votes(label) = votes(label) + 1;
        
    end
end

[~, assignedClass] = max(votes);
end
