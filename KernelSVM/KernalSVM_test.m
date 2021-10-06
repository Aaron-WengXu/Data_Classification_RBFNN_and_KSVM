clc;
clear;
X = load('data_train.mat');
Y = load('label_train.mat');
newX = load('data_test');

%% Tune hyperparameters using k-fold cross validation method   

% Kernel Function in fitcsvm
% Gaussian:   K(x_i, x_j) = exp(-||x_i - x_j||^2)
% Scaled Kernel Function: K(x_i, x_j) = exp(-||x_i - x_j||^2/theta^2)
%
% Gaussian function in 7602 Slides:
% K(x_i, x_j) = exp(-||x_i - x_j||^2/(2theta^2))
%
%
% Hyper-parameter to be tuned
% Kernel Scale and Box Constraint
theta0_exp = -10:1:4;
theta0 = 2.^theta0_exp; % [1] CW Hsu,a comparison of methods for multiclass support vector machines
theta = theta0*2^0.5;
% theta = 0.5:0.5:10;
C_exp = -2:1:12;
C = 2.^C_exp;

% k-fold cross validation
best_acc = 0;   % Initialization

% Requried for consistent random number stream
s = RandStream('mcg16807','Seed',0);
RandStream.setGlobalStream(s);

cv_part = cvpartition(Y.label_train,'KFold',10);    % Create indices for training/validation subsets

% Test every configuration
   
for p1 = 1:numel(theta)
    TestModelParameters.theta = theta(p1);
    for p2 = 1:numel(C)
        TestModelParameters.C = C(p2);
        val_acc = zeros(10,1);   % Initialisation
        for k = 1:10
            % Collect training/validation sets
            val_trainX = X.data_train(cv_part.training(k),:);
            val_trainY = Y.label_train(cv_part.training(k),:);
            
            val_testX = X.data_train(cv_part.test(k),:);
            val_testY = Y.label_train(cv_part.test(k),:);
            
            %         % Data Normalization
            %         mean_X = mean(val_trainX,1);
            %         std_X = std(val_trainX);
            %         std_X(std_X==0) = 1e-4;                 % For numerical stability
            %         val_trainX = bsxfun(@rdivide,val_trainX-repmat(mean_X,size(val_trainX,1),1),std_X);
            %         val_testX = bsxfun(@rdivide,val_testX-repmat(mean_X,size(val_testX,1),1),std_X);
            
            % Training and Testing
            [~,val_acc(k)] = KSVM(val_trainX,val_trainY,val_testX,val_testY,TestModelParameters);
        end
        
        % Average the validation accuracy
        ValAcc = mean(val_acc);
        
        % Check if current configuration is the best
        if ValAcc > best_acc
            best_acc = ValAcc;
            best_theta = theta(p1);
            best_theta0 = theta0(p1);
            best_C = C(p2);
        end
    end
end

% Use the best settings
ModelParameters.theta = best_theta;
ModelParameters.C = best_C;

% Evaluation
[SVMModel,label,TrainingAcc] = KSVM_test(X.data_train,Y.label_train,newX.data_test,ModelParameters);  


