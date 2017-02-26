close all ; clear all;
cd /Users/deepaliKamat/Documents/MATLAB/hwk6files_forMycourses
addpath /Users/deepaliKamat/Documents/MATLAB/hwk6files_forMycourses/libsvm-3.18/matlab/

load('ex6data2.mat');  %load Andrew Ng data
%convert from y=[0,1} to y={1,2} for confusion matrix indexing
y(y==1)=2;
y(y==0)=1;
 
options.method = 'SVM';
options.numberOfFolds = 5;
options.svm_t=2;
options.svm_c=1;
options.svm_g=200;
acc = -5;
for c = 1:2:100
    for g = 0:10:300
        options.svm_c = c;
        options.svm_g= g;
        
        [confusionMatrix,accuracy] =  classify677_hwk6(X,y,options);
        if (accuracy> acc)
            acc = accuracy;
            finalc= c;
            finalg = g;
        end
        
    end
end

options.svm_c=finalc;
options.svm_g=finalg;
[confusionMatrix,accuracy] =  classify677_hwk6(X,y,options);

% Find Indices of Positive and Negative Examples
pos = find(y == 2); neg = find(y == 1);
% Plot Examples
hold off
plot(X(pos, 1), X(pos, 2), 'g+','LineWidth', 3, 'MarkerSize', 12)
hold on;
plot(X(neg, 1), X(neg, 2), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 7)
 
str = sprintf('The best accuracy is: %0.2f%%:c=%d, g=%d\n',acc*100,finalc,finalg);
title(str,'fontsize',14);


