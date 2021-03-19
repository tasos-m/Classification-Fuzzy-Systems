%% Classification - part 2
%% Import and split data

clear;
close all;
data = importdata('data.csv');
data = data.data;
preproc = 1;
[dataTrn,dataVal,dataChk]=split_scale(data,preproc);
[idx,weights] = relieff(dataTrn(:,1:end-1),dataTrn(:,end),10);

% Number of feautures
NF = [3 5 8 10];

% Radii
R = [0.3 0.5 0.8];

err = zeros(length(NF),length(R));

%% Grid Search & 5 Fold Cross Validation
for nf = 1:length(NF)
    for r = 1:length(R)
        for k=1:5
            % Split the dataset in 5 folds
            indices = crossvalind('Kfold',length(dataTrn),5);            
            % Choose the best feautures
            dataReduced = [dataTrn(:,idx(1:NF(nf))) dataTrn(:,end)];
            % Choose the k-th fold to be the validation data for this round
            val =  (indices == k);
            % Training, and validation indices
            indVal = find(val == 1);
            indTrain = find(val == 0);
            
            train80 = dataReduced(indTrain,:); 
            val20 = dataReduced(indVal,:);

            % Clustering Per Class
            radius = R(r);
            [c1,sig1] = subclust(train80(train80(:,end)==1,:),radius);
            [c2,sig2] = subclust(train80(train80(:,end)==2,:),radius);
            [c3,sig3] = subclust(train80(train80(:,end)==3,:),radius);
            [c4,sig4] = subclust(train80(train80(:,end)==4,:),radius);
            [c5,sig5] = subclust(train80(train80(:,end)==5,:),radius);
            
            num_rules = size(c1,1)+size(c2,1)+size(c3,1)+size(c4,1)+size(c5,1);

            %Build FIS From Scratch
            fis = newfis('FIS_SC','sugeno');

            %Add Input-Output Variables
            for i = 1:size(train80,2)-1
                  names_in(i) = sprintf("sth%d",i);
            end
            for i = 1:size(train80,2)-1
                fis = addvar(fis,'input',names_in{i},[0 1]);
            end
            fis = addvar(fis,'output','out1',[1 5]);

            %Add Input Membership Functions
            name = 'sth';
            for i = 1:size(train80,2)-1
                for j = 1:size(c1,1)
                    fis = addmf(fis,'input',i,name,'gaussmf',[sig1(i) c1(j,i)]); 
                end
                for j = 1:size(c2,1)
                    fis = addmf(fis,'input',i,name,'gaussmf',[sig2(i) c2(j,i)]);
                end
                for j = 1:size(c3,1)
                    fis = addmf(fis,'input',i,name,'gaussmf',[sig3(i) c3(j,i)]);
                end
                for j = 1:size(c4,1)
                    fis = addmf(fis,'input',i,name,'gaussmf',[sig4(i) c4(j,i)]);
                end
                for j = 1:size(c5,1)
                    fis = addmf(fis,'input',i,name,'gaussmf',[sig5(i) c5(j,i)]);
                end                
            end

            %Add Output Membership Functions
            params = [ones(1,size(c1,1)) 2*ones(1,size(c2,1)) 3*ones(1,size(c3,1)) 4*ones(1,size(c4,1)) 5*ones(1,size(c5,1))];  
            for i = 1:num_rules
                fis = addmf(fis,'output',1,name,'constant',params(i));
            end

            %Add FIS Rule Base
            ruleList = zeros(num_rules,size(train80,2));
            for i=1:size(ruleList,1)
                ruleList(i,:) = i;
            end
            ruleList = [ruleList ones(num_rules,2)];
            fis = addrule(fis,ruleList);
            
            % Train & Evaluate ANFIS
            [trnFis,trnError,~,valFis,valError]=anfis(train80,fis,[100 0 0.01 0.9 1.1],[],val20);
           

            err(nf, r) = err(nf, r) + mean(valError);
        end
        err(nf, r) = err(nf, r) / 5;
    end
end

%% Error plots for the models
% Mean error respective to number of featuresfigure
hold on
grid on
title('Mean error respective to number of features');
xlabel('Number of features')
ylabel('RMSE')
plot(NF, err(:, 1));
plot(NF, err(:, 2));
plot(NF, err(:, 3));
legend('0.3 radius', '0.5 radius', '0.8 radius');

% Mean error respective to the cluster radius
figure
hold on
grid on
title('Mean error respective to the cluster radius');
xlabel('Cluster radius')
ylabel('RMSE')
plot(R, err(1, :));
plot(R, err(2, :));
plot(R, err(3, :));
plot(R, err(4, :));
legend('3 features', '5 features', '8 features','10 features');

%% Find the best model
[bestNF, bestNR] = find(err==min(err(:)));
disp("Best model:");
disp("     NF    NR");
disp([NF(bestNF), R(bestNR)]);

%% Data after selecting the best number of feautures
dataFinal = data(:,idx(1:NF(bestNF)));
trainFinal = [dataTrn(:,idx(1:NF(bestNF))) dataTrn(:,end)];
valFinal   = [dataVal(:,idx(1:NF(bestNF))) dataVal(:,end)];
chkFinal   = [dataChk(:,idx(1:NF(bestNF))) dataChk(:,end)];
%% Final Model
% Clustering Per Class
radius = R(bestNR); %% the best radius
[c1,sig1] = subclust(trainFinal(trainFinal(:,end)==1,:),radius);
[c2,sig2] = subclust(trainFinal(trainFinal(:,end)==2,:),radius);
[c3,sig3] = subclust(trainFinal(trainFinal(:,end)==3,:),radius);
[c4,sig4] = subclust(trainFinal(trainFinal(:,end)==4,:),radius);
[c5,sig5] = subclust(trainFinal(trainFinal(:,end)==5,:),radius);

num_rules = size(c1,1)+size(c2,1)+size(c3,1)+size(c4,1)+size(c5,1);

%Build FIS From Scratch
fis = newfis('FIS_SC','sugeno');

%Add Input-Output Variables
for i = 1:size(trainFinal,2)-1
      names_in(i) = sprintf("sth%d",i);
end
for i = 1:size(trainFinal,2)-1
    fis = addvar(fis,'input',names_in{i},[0 1]);
end
fis = addvar(fis,'output','out1',[1 5]); 

%Add Input Membership Functions
name = 'sth';
for i = 1:size(trainFinal,2)-1
    for j = 1:size(c1,1)
        fis = addmf(fis,'input',i,name,'gaussmf',[sig1(i) c1(j,i)]); 
    end
    for j = 1:size(c2,1)
        fis = addmf(fis,'input',i,name,'gaussmf',[sig2(i) c2(j,i)]);
    end
    for j = 1:size(c3,1)
        fis = addmf(fis,'input',i,name,'gaussmf',[sig3(i) c3(j,i)]);
    end
    for j = 1:size(c4,1)
        fis = addmf(fis,'input',i,name,'gaussmf',[sig4(i) c4(j,i)]);
    end
    for j = 1:size(c5,1)
        fis = addmf(fis,'input',i,name,'gaussmf',[sig5(i) c5(j,i)]);    
    end
end
%Add Output Membership Functions
params = [ones(1,size(c1,1)) 2*ones(1,size(c2,1)) 3*ones(1,size(c3,1)) 4*ones(1,size(c4,1)) 5*ones(1,size(c5,1))];  
for i = 1:num_rules
    fis = addmf(fis,'output',1,name,'constant',params(i));
end

%Add FIS Rule Base
ruleList=zeros(num_rules,size(trainFinal,2));
for i=1:size(ruleList,1)
    ruleList(i,:) = i;
end
ruleList = [ruleList ones(num_rules,2)];
fis = addrule(fis,ruleList);

% Train & Evaluate ANFIS
[trnFis,trnError,~,valFis,valError] = anfis(trainFinal,fis,[100 0 0.01 0.9 1.1],[],valFinal);
Y = evalfis(chkFinal(:,1:end-1),valFis);
Y = round(Y);

%% Plots and Metrics for the final model
%% Learning Curve
figure;
plot([trnError valError],'LineWidth',2); grid on;
legend('Training Error','Validation Error');
xlabel('# of Epochs');
ylabel('Error');
%% Plots of predictions and real values
figure;
hold on
title('Predictions and Real values');
xlabel('Test dataset sample')
ylabel('y')
plot(1:length(Y), Y,'o','Color','blue');
plot(1:length(Y), chkFinal(:, end),'o','Color','red');
legend('Predictions', 'Real values');
%% Plot the Membership Functions 
for i = 1:size(trainFinal,2)-1
    
    figure
    plotmf(fis, 'input',i);
    
    figure
    plotmf(trnFis, 'input',i);
end
%% Error matrix
classes = unique(data(:, end));
dim = length(classes);
errorMatrix = zeros(dim);
N = length(chkFinal);
for i = 1:N
    xpos = find(classes == Y(i));
    ypos = find(classes == data(i, end));
    errorMatrix(xpos, ypos) = errorMatrix(xpos, ypos) + 1;
end
%% Calculate Metrics
OA = trace(errorMatrix) / N;
sum_of_rows = sum(errorMatrix, 2);
sum_of_cols = sum(errorMatrix, 1);
PA = zeros(1, dim);
UA = zeros(1, dim);
for i = 1:dim
    PA(i) = errorMatrix(i, i) / sum_of_cols(i);
    UA(i) = errorMatrix(i, i) / sum_of_rows(i);
end
k = (N^2 * OA - PA .* UA) / (N^2 - PA .* UA); 