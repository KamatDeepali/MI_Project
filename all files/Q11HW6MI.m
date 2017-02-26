close all ; clear all;clc;
addpath /Users/deepaliKamat/Documents/MATLAB/hwk6files_forMycourses/libsvm-3.18/matlab/

load('ex6data2.mat');  
pos = find(y == 1); neg = find(y == 0);
% Plot Examples
hold off
plot(X(pos, 1), X(pos, 2), 'g+','LineWidth', 3, 'MarkerSize', 12)
hold on;
plot(X(neg, 1), X(neg, 2), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 7)
 
% Cost value
C = 1;
G = 0;
% Call svmtrain from LIBSVM
for G = 0:10:2000
    %eval(['model = svmtrain(y,X,''-t 2 -c 1 -g 90 ' num2str(G) ''');']);
    eval(['model = svmtrain(y,X,''-t  ' num2str(2)  ' -c ' num2str(C) ' -g ' num2str(G) ''' );']);
    

    ytest = y ;
    predict = svmpredict( y, X, model, '-q');
    s = sum(abs(predict-y));
    if s==0
        break
   end
end
 visualizeBoundary2D(X,y,model) 
str = sprintf('g=%d',G);
title(str,'fontsize',14);
print -dpng hwk6_q11.png
