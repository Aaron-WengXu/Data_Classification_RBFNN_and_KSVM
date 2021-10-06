function TestingAcc = PredictKSVM(Model,test_X,test_Y)

[label,~] = predict(Model,test_X);

TestingAcc = ComputeAcc(test_Y,label);

end