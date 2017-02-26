close all ; clear all;
cd /Users/deepaliKamat/Documents/MATLAB/hwk6files_forMycourses
addpath /Users/deepaliKamat/Documents/MATLAB/hwk6files_forMycourses/libsvm-3.18/matlab/

load('ex6data2.mat');  
%load Andrew Ng data
% Find Indices of Positive and Negative Examples
pos = find(y == 1); neg = find(y == 0);
 
% Plot Examples
hold off
plot(X(pos, 1), X(pos, 2), 'g+','LineWidth', 3, 'MarkerSize', 12)
hold on;
plot(X(neg, 1), X(neg, 2), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 7)
 
% Cost value
C = 1;
 
% Call svmtrain from LIBSVM
%   -t 0 says to do a linear kernel
%   -c <value>  sets cost to <value>, higher c means SVM will weigh errors more
eval(['model = svmtrain(y,X,''-t 0 -c ' num2str(C) ''');']);
 
% SVM solves wx+b
% alpha values are stored in model.sv_coeff
% Note: alpha values are really y(i) * alpha(i)- so no need to include y when solving for w
% support vectors are stored in model.SVs
% solve for w here:
% w = <insert code here>
w = (model.sv_coef' * full(model.SVs));  %full converts from sparse to full matrix representation
% b is stored in -model.rho
b = -model.rho;
 
%plot boundary ontop of data
xp = linspace(min(X(:,1)), max(X(:,1)), 100);
yp = - (w(1)*xp + b)/w(2);
hold on;
plot(xp, yp, 'b:','linewidth',3); 
 
%Predictions are sign(<x,w> + b), note: can do all predictions in one line
%predictionsTrain = <insert code here>
predictionsTrain = sign(X * w' + b);
predictionsTrain(predictionsTrain==-1) = 0;  %change -1 to 0 to match GT
 
%compute training error
%predictionsTrainError = <insert code here>
predictionsTrainError = sum(predictionsTrain~= y)/length(y);
fprintf('Error on train set = %0.2f%%\n',predictionsTrainError*100);
 
%Now we will see how this does on a test set
Xtest = [ 1 3; 2 3; 3 3; 4 3; 1 4; 2 4; 3 4; 4 4];
ytest = [0 0 0 1 0 1 1 1]';
%predictionsTest = <insert code here>
predictionsTest = sign(Xtest * w' + b);
predictionsTest(predictionsTest==-1) = 0;  %change -1 to 0 to match GT
%predictionsTest = <insert code here>
predictionsTestError = sum(predictionsTest~= ytest)/length(ytest);
fprintf('Error on test set = %0.2f%%\n',predictionsTestError*100);
%print -dpng hwk6_q5.png

