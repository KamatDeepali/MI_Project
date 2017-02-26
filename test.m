%load ('final_letter_data.csv');

addpath /Users/deepaliKamat/Documents/MATLAB/hwk6files_forMycourses/libsvm-3.18/matlab
addpath /Users/deepaliKamat/Documents/MATLAB/hwk6files_forMycourses

formatSpec = '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f';

T = readtable('final_letter_data.txt','Delimiter','\t', ...
    'Format', formatSpec);

y = T(:,1);
X = T(:, 2:end);

C = 1;
g = 300;

eval(['model = svmtrain(y,X,''-t 0 -c ' num2str(C) ' -g ' num2str(g) ''' );']);

predict = svmpredict(y, X, model);


        
%fprintf('The best accuracy is: %0.2f%%:c=%d, g=%d\n',maxAccuracy*100,best_c,best_g);
%The best accuracy is: 99.65%:c=3, g=30
% Confusion Matrix:
%    382      1 
%      2    478 
 
%plot best combination of c and g
% Find Indices of Positive and Negative Examples
%convert from y={1,2} to y={0,1} for plotting


