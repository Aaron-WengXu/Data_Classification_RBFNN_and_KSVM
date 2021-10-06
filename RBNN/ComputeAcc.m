function acc = ComputeAcc(test_Y,label)

if length(test_Y)~= length(label)
    error('The length of testing data is not consistent with the length of its label');
end
L = length(label);
CorrectSum = 0;

for i = 1:L
    if test_Y(i)== label(i)
        CorrectSum = CorrectSum + 1;        
    end    
end

acc = CorrectSum / L;

end