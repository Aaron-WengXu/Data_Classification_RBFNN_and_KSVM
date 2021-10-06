function [Model,TestingAcc]  = RBNN(trainX,trainY,testX,testY,TestModelParameters)

% Requried for consistency
s = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(s);

% Train a RBNN
Model = newrb(trainX',trainY',0,TestModelParameters.spread,TestModelParameters.MN,25);
% Model = fitcsvm(trainX,trainY,'KernelFunction','rbf',...
%        'Standardize',false,'BoxConstraint',Inf,'KernelScale',TestModelParameters.theta,'ClassNames',[-1,1]);   
% Test the RBNN
TestingAcc = PredictSim(Model,testX,testY);

end