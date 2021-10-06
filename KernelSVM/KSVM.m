function [Model,TestingAcc]  = KSVM(trainX,trainY,testX,testY,TestModelParameters)

% Requried for consistency
s = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(s);

% Train a SVM
Model = fitcsvm(trainX,trainY,'KernelFunction','rbf',...
       'Standardize',false,'BoxConstraint',TestModelParameters.C,'KernelScale',TestModelParameters.theta,'ClassNames',[-1,1]);
% Model = fitcsvm(trainX,trainY,'KernelFunction','rbf',...
%        'Standardize',false,'BoxConstraint',Inf,'KernelScale',TestModelParameters.theta,'ClassNames',[-1,1]);   
% Test the SVM
TestingAcc = PredictKSVM(Model,testX,testY);

end