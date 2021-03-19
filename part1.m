%% Classification - part 1 
%% Import and split data
clear;
close all;
data = importdata('haberman.data');
 
preproc = 1;
[dataTrn,dataVal,dataChk] = split_scale(data,preproc);

%% First Model 
% FIS generation options
optFis1 = genfisOptions('SubtractiveClustering','ClusterInfluenceRange', 0.2);
% Generate fuzzy model from data
model_fis1 = genfis(dataTrn(:,1:end-1),dataTrn(:,end), optFis1);

for i = 1:length(model_fis1.output.mf)
   model_fis1.output.mf(i).type = 'constant';
   model_fis1.output.mf(i).params = model_fis1.output.mf(i).params(end); 
end

% Training options
opt_of_train1 = anfisOptions;
opt_of_train1.InitialFIS = model_fis1;
opt_of_train1.ValidationData = dataVal;
opt_of_train1.EpochNumber = 100;   

% Train the model
[trnFis1,trnError1,~,valFis1,valError1] = anfis(dataTrn, opt_of_train1);

% Evaluate the fuzzy model
Y_1 = evalfis(dataChk(:,1:end-1),valFis1);
Y_1 = round(Y_1);

% Learning Curve
figure;
plot([trnError1 valError1],'LineWidth',2); grid on;
legend('Training Error','Validation Error');
xlabel('# of Epochs');
ylabel('Error');
% Plot Membership Functions
for i = 1:length(trnFis1.input)
    figure
    plotmf(trnFis1, 'input', i);
end
% Error matrix
classes1 = unique(data(:, end));
dim_1 = length(classes1);
errorMatrix1 = zeros(dim_1);
N_1 = length(dataChk);
for i = 1:N_1
    xpos_1 = find(classes1 == Y_1(i));
    ypos_1 = find(classes1 == data(i, end));
    errorMatrix1(xpos_1, ypos_1) = errorMatrix1(xpos_1, ypos_1) + 1;
end

% Calculate metrics
OA_1 = trace(errorMatrix1) / N_1;
sum_of_rows1 = sum(errorMatrix1, 2);
sum_of_cols1 = sum(errorMatrix1, 1);
PA_1 = zeros(1, dim_1);
UA_1 = zeros(1, dim_1);
for i = 1:dim_1
    PA_1(i) = errorMatrix1(i, i) / sum_of_cols1(i);
    UA_1(i) = errorMatrix1(i, i) / sum_of_rows1(i);
end
k_1 = (N_1^2 * OA_1 - PA_1 .* UA_1) / (N_1^2 - PA_1 .* UA_1); 
%% Second Model
% FIS generation options
optFis2 = genfisOptions('SubtractiveClustering','ClusterInfluenceRange', 0.8);

% Generate fuzzy model from data
model_fis2 = genfis(dataTrn(:,1:end-1),dataTrn(:,end), optFis2);

for i = 1:length(model_fis2.output.mf)
   model_fis2.output.mf(i).type = 'constant';
   model_fis2.output.mf(i).params = model_fis2.output.mf(i).params(end); 
end

% Training options
opt_of_train2 = anfisOptions;
opt_of_train2.InitialFIS = model_fis2;
opt_of_train2.ValidationData = dataVal;
opt_of_train2.EpochNumber = 100;    

% Train the model
[trnFis2,trnError2,~,valFis2,valError2] = anfis(dataTrn, opt_of_train2);

% Evaluate the fuzzy model
Y_2 = evalfis(dataChk(:,1:end-1),valFis2);
Y_2 = round(Y_2);

% Learning Curve
figure;
plot([trnError2 valError2],'LineWidth',2); grid on;
legend('Training Error','Validation Error');
xlabel('# of Epochs');
ylabel('Error');
% Plot Membership Functions
for i = 1:length(trnFis2.input)
    figure
    plotmf(trnFis2, 'input', i);
end
% Error matrix
classes2 = unique(data(:, end));
dim_2 = length(classes2);
errorMatrix2 = zeros(dim_2);
N_2 = length(dataChk);
for i = 1:N_2
    xpos_2 = find(classes2 == Y_2(i));
    ypos_2 = find(classes2 == data(i, end));
    errorMatrix2(xpos_2, ypos_2) = errorMatrix2(xpos_2, ypos_2) + 1;
end

% Calculate metrics
OA_2 = trace(errorMatrix2) / N_2;
sum_of_rows2 = sum(errorMatrix2, 2);
sum_of_cols2 = sum(errorMatrix2, 1);
PA_2 = zeros(1, dim_2);
UA_2 = zeros(1, dim_2);
for i = 1:dim_2
    PA_2(i) = errorMatrix2(i, i) / sum_of_cols2(i);
    UA_2(i) = errorMatrix2(i, i) / sum_of_rows2(i);
end
k_2 = (N_2^2 * OA_2 - PA_2 .* UA_2) / (N_2^2 - PA_2 .* UA_2); 
%% Third Model 
% Clustering Per Class
radius_3 = 0.2;
[c1_3,sig1_3] = subclust(dataTrn(dataTrn(:,end)==1,:),radius_3);
[c2_3,sig2_3] = subclust(dataTrn(dataTrn(:,end)==2,:),radius_3);
num_rules_3 = size(c1_3,1)+size(c2_3,1);

%Build FIS From Scratch
fis3 = newfis('FIS_SC','sugeno');

%Add Input-Output Variables
names_in = {'in1','in2','in3'};
for i = 1:size(dataTrn,2)-1
    fis3 = addvar(fis3,'input',names_in{i},[0 1]);
end
fis3 = addvar(fis3,'output','out1',[1 2]);

%Add Input Membership Functions
name = 'sth';
for i = 1:size(dataTrn,2)-1
    for j = 1:size(c1_3,1)
        fis3 = addmf(fis3,'input',i,name,'gaussmf',[sig1_3(i) c1_3(j,i)]); 
    end
    for j = 1:size(c2_3,1)
        fis3 = addmf(fis3,'input',i,name,'gaussmf',[sig2_3(i) c2_3(j,i)]);
    end
end

%Add Output Membership Functions
params_3 = [ones(1,size(c1_3,1)) 2*ones(1,size(c2_3,1))];  
for i = 1:num_rules_3
    fis3 = addmf(fis3,'output',1,name,'constant',params_3(i));
end

%Add FIS Rule Base
ruleList_3=zeros(num_rules_3,size(dataTrn,2));
for i=1:size(ruleList_3,1)
    ruleList_3(i,:) = i;
end
ruleList_3 = [ruleList_3 ones(num_rules_3,2)];
fis3 = addrule(fis3,ruleList_3);

% Train & Evaluate ANFIS
[trnFis3,trnError3,~,valFis3,valError3]=anfis(dataTrn,fis3,[100 0 0.01 0.9 1.1],[],dataVal);
Y_3=evalfis(dataChk(:,1:end-1),valFis3);
Y_3=round(Y_3);
% Learning Curve
figure;
plot([trnError3 valError3],'LineWidth',2); grid on;
legend('Training Error','Validation Error');
xlabel('# of Epochs');
ylabel('Error');
% Plot Membership Functions
for i = 1:length(trnFis3.input)
    figure
    plotmf(trnFis3, 'input', i);
end
% Error matrix
classes3 = unique(data(:, end));
dim_3 = length(classes3);
errorMatrix3 = zeros(dim_3);
N_3 = length(dataChk);
for i = 1:N_3
    xpos_3 = find(classes3 == Y_3(i));
    ypos_3 = find(classes3 == data(i, end));
    errorMatrix3(xpos_3, ypos_3) = errorMatrix3(xpos_3, ypos_3) + 1;
end

% Calculate metrics
OA_3 = trace(errorMatrix3) / N_3;
sum_of_rows3 = sum(errorMatrix3, 2);
sum_of_cols3 = sum(errorMatrix3, 1);
PA_3 = zeros(1, dim_3);
UA_3 = zeros(1, dim_3);
for i = 1:dim_3
    PA_3(i) = errorMatrix3(i, i) / sum_of_cols3(i);
    UA_3(i) = errorMatrix3(i, i) / sum_of_rows3(i);
end
k_3 = (N_3^2 * OA_3 - PA_3 .* UA_3) / (N_3^2 - PA_3 .* UA_3); 
%% Fourth model
% Clustering Per Class
radius_4 = 0.8;
[c1_4,sig1_4] = subclust(dataTrn(dataTrn(:,end)==1,:),radius_4);
[c2_4,sig2_4] = subclust(dataTrn(dataTrn(:,end)==2,:),radius_4);
num_rules_4 = size(c1_4,1)+size(c2_4,1);

%Build FIS From Scratch
fis4 = newfis('FIS_SC','sugeno');

%Add Input-Output Variables
names_in = {'in1','in2','in3'};
for i = 1:size(dataTrn,2)-1
    fis4 = addvar(fis4,'input',names_in{i},[0 1]);
end
fis4 = addvar(fis4,'output','out1',[1 2]); 

%Add Input Membership Functions
name = 'sth';
for i = 1:size(dataTrn,2)-1
    for j = 1:size(c1_4,1)
        fis4 = addmf(fis4,'input',i,name,'gaussmf',[sig1_4(i) c1_4(j,i)]); 
    end
    for j = 1:size(c2_4,1)
        fis4 = addmf(fis4,'input',i,name,'gaussmf',[sig2_4(i) c2_4(j,i)]);
    end
end

%Add Output Membership Functions
params_4=[ones(1,size(c1_4,1)) 2*ones(1,size(c2_4,1))];
for i = 1:num_rules_4
    fis4 = addmf(fis4,'output',1,name,'constant',params_4(i));
end

%Add FIS Rule Base
ruleList_4=zeros(num_rules_4,size(dataTrn,2));
for i=1:size(ruleList_4,1)
    ruleList_4(i,:) = i;
end
ruleList_4 = [ruleList_4 ones(num_rules_4,2)];
fis4 = addrule(fis4,ruleList_4);

% Train & Evaluate ANFIS
[trnFis4,trnError4,~,valFis4,valError4]=anfis(dataTrn,fis4,[100 0 0.01 0.9 1.1],[],dataVal);
Y_4=evalfis(dataChk(:,1:end-1),valFis4);
Y_4=round(Y_4);

% Learning Curve
figure;
plot([trnError4 valError4],'LineWidth',2); grid on;
legend('Training Error','Validation Error');
xlabel('# of Epochs');
ylabel('Error');
% Plot Membership Functions
for i = 1:length(trnFis4.input)
    figure
    plotmf(trnFis4, 'input', i);
end

% Error matrix
classes4 = unique(data(:, end));
dim_4 = length(classes4);
errorMatrix4 = zeros(dim_4);
N_4 = length(dataChk);
for i = 1:N_4
    xpos_4 = find(classes4 == Y_4(i));
    ypos_4 = find(classes4 == data(i, end));
    errorMatrix4(xpos_4, ypos_4) = errorMatrix4(xpos_4, ypos_4) + 1;
end

% Calculate metrics
OA_4 = trace(errorMatrix4) / N_4;
sum_of_rows4 = sum(errorMatrix4, 2);
sum_of_cols4 = sum(errorMatrix4, 1);
PA_4 = zeros(1, dim_4);
UA_4 = zeros(1, dim_4);
for i = 1:dim_4
    PA_4(i) = errorMatrix4(i, i) / sum_of_cols4(i);
    UA_4(i) = errorMatrix4(i, i) / sum_of_rows4(i);
end
k_4 = (N_4^2 * OA_4 - PA_4 .* UA_4) / (N_4^2 - PA_4 .* UA_4); 