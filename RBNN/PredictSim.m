function TestingAcc = PredictSim(Model,test_X,test_Y)

label = sim(Model,test_X');

for i = 1:size(label,2)
    if label(i)>=0
        label(i) = 1;
    else
        label(i) = -1;
    end
end

TestingAcc = ComputeAcc(test_Y',label);

end