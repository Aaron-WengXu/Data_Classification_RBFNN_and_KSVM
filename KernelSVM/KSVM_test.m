function [Model,label,TrainingAcc]  = KSVM_test(trainX,trainY,testX,TestModelParameters)

% Requried for consistency
s = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(s);

%Train a SVM
Model = fitcsvm(trainX,trainY,'KernelFunction','rbf',...
       'Standardize',false,'BoxConstraint',TestModelParameters.C,'KernelScale',TestModelParameters.theta,'ClassNames',[-1,1]);
% Model = fitcsvm(trainX,trainY,'KernelFunction','rbf',...
%        'Standardize',false,'BoxConstraint',Inf,'KernelScale',TestModelParameters.theta,'ClassNames',[-1,1]);
% Calculate Traning Accuracy
TrainingAcc = PredictKSVM(Model,trainX,trainY);
% Test the SVM
[label,~] = predict(Model,testX);
end