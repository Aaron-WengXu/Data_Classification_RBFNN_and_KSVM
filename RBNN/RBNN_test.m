function [Model,label,TrainingAcc]  = RBNN_test(trainX,trainY,testX,TestModelParameters)

% Requried for consistency
s = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(s);

% Train a SVM
Model = newrb(trainX',trainY',0,TestModelParameters.spread,TestModelParameters.MN,25);

% Calculate Traning Accuracy
TrainingAcc = PredictSim(Model,trainX,trainY);

% Test the RBNN
label = sim(Model,testX');
for i = 1:size(label,2)
    if label(i)>=0
        label(i) = 1;
    else
        label(i) = -1;
    end
end

end