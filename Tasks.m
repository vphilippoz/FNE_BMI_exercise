close all; clc; clear;

init

%% ----- TASK 1 -----
disp('----- TASK 1 -----')

nb_folds = length(parms.patterns.MI_neurons);

for fold = 1:nb_folds
    figure;
    data = parms.patterns.MI_neurons{fold}';
    data_mean = mean(data,2);
    [~, idxs] = sort(data_mean, 1,"descend");

    fig = imagesc(data(idxs,:));
    set(gca,'XTickLabel',parms.patterns.time{fold}(1:200:end))
    title(['Fold ' num2str(fold)])
    xlabel('Time [s]') 
    ylabel('Neurons')

    saveas(fig,['Plots/task1_MI_neurons_' num2str(fold) '.png'])
end
close all;

%% ----- TASK 2 -----
disp('----- TASK 2 -----')
train_folds = 1;
test_folds = 10;

set_train(train_folds); % Train datasets
set_test(test_folds); % Test datasets
set_type('theta') % Type of prediction

train; % Trains the model

%% ----- TASK 3 -----
disp('----- TASK 3 -----')

newwindow(1, 'train');
newwindow(2, 'test');
close all;

%% ----- TASK 4 -----
disp('----- TASK 4 -----')

% train_perf = zeros(10,2);
% test_perf = zeros(10,2);
% train_fold = zeros(10,1);
% test_fold = zeros(10,1);
% 
% for i = 1:10
%     train_folds = round(9*rand()+1);
%     test_folds = round(9*rand()+1);
%     while train_folds == test_folds
%         test_folds = round(9*rand()+1);
%     end
%     train_fold(i) = train_folds;
%     test_fold(i) = test_folds;
% 
%     set_train(train_folds);
%     set_test(test_folds);
%     set_type('theta');
%     
%     train;
%     
%     train_perf(i,:) = parms.net.fvaf_train;
%     test_perf(i,:) = parms.net.fvaf_test;
% end
% minnn = mean(all_values,1);
% all_values = [train_fold test_fold round(train_perf,3) round(test_perf,3)];

% train_perf = zeros(10,2);
% test_perf = zeros(10,2);
% train_fold = zeros(10,2);
% test_fold = zeros(10,1);
% 
% for i = 1:10
%     train_folds = round(9*rand(1,2)+1);
%     test_folds = round(9*rand()+1);
%     while isempty(intersect(train_folds,test_folds)) == false 
%         test_folds = round(9*rand()+1);
%     end
%     train_fold(i,:) = train_folds;
%     test_fold(i,:) = test_folds;
% 
%     set_train(train_folds);
%     set_test(test_folds);
%     set_type('theta');
%     
%     train;
%     
%     train_perf(i,:) = parms.net.fvaf_train;
%     test_perf(i,:) = parms.net.fvaf_test;
% end
% means = round([mean(train_perf,1) mean(test_perf,1)],3);
% all_values = [train_fold test_fold round(train_perf,3) round(test_perf,3)];
% all_values = [all_values;zeros(1, 3) means];

nb_repeats = 10;
mean_perfs = zeros(8,5);
train_perfs = zeros(8,nb_repeats,2);
test_perfs = zeros(8,nb_repeats,2);


for nb_folds = 1:8
    disp(nb_folds)
    train_perf = zeros(nb_repeats,2);
    test_perf = zeros(nb_repeats,2);
    train_fold = zeros(nb_repeats,nb_folds);
    test_fold = zeros(nb_repeats,1);
    for i = 1:nb_repeats
        train_folds = round(9*rand(1,nb_folds)+1);
        test_folds = round(9*rand()+1);
        while length(unique([train_folds test_folds])) ~= length([train_folds test_folds])
            train_folds = round(9*rand(1,nb_folds)+1);
            test_folds = round(9*rand()+1);
        end
        train_fold(i,:) = train_folds;
        test_fold(i,:) = test_folds;
    
        set_train(train_folds);
        set_test(test_folds);
        set_type('theta');
        
        train;
        
        train_perf(i,:) = parms.net.fvaf_train;
        test_perf(i,:) = parms.net.fvaf_test;
    end
    train_perfs(nb_folds,:,:) = train_perf;
    test_perfs(nb_folds,:,:) = test_perf;
    mean_perfs(nb_folds,:) = round([nb_folds mean(train_perf,1) mean(test_perf,1)],3);
end

hold on;
fig = plot(mean_perfs(:,1),mean_perfs(:,2),'color',[1 0 0]); % 'Train - Shoulder'
plot(mean_perfs(:,1),mean_perfs(:,3),'color',[0.5 0 0]) % 'Train - Elbow'
plot(mean_perfs(:,1),mean_perfs(:,4),'color',[0 0 1]) % 'Test - Shoulder'
plot(mean_perfs(:,1),mean_perfs(:,5),'color',[0 0 0.5]) % 'Test - Elbow'
xlabel('Fold quantity')
ylabel('FVAF')
title(['Performance as a function of training set size (' num2str(nb_repeats) ' repeats)'])
legend({'Train - Shoulder','Train - Elbow', 'Test - Shoulder', 'Test - Elbow'},'Location','best')
saveas(fig,['Plots/task4_rep' num2str(nb_repeats) '.png'])
close all;

%% ----- TASK 5 -----
disp('----- TASK 5 -----')

% train_perf = zeros(10,2);
% test_perf = zeros(10,2);
% train_fold = zeros(10,2);
% test_fold = zeros(10,1);
% 
% for i = 1:10
%     train_folds = round(9*rand(1,2)+1);
%     test_folds = round(9*rand()+1);
%     while length(unique([train_folds test_folds])) ~= length([train_folds test_folds])
%         train_folds = round(9*rand(1,2)+1);
%         test_folds = round(9*rand()+1);
%     end
%     train_fold(i,:) = train_folds;
%     test_fold(i,:) = test_folds;
% 
%     set_train(train_folds);
%     set_test(test_folds);
%     set_type('X');
%     
%     train;
%     
%     train_perf(i,:) = parms.net.fvaf_train;
%     test_perf(i,:) = parms.net.fvaf_test;
% end
% means = round(mean([train_perf test_perf],1),3);
% all_values = [train_fold test_fold round(train_perf,3) round(test_perf,3)];
% all_values = [all_values; zeros(1, 3) means];

types = {'X', 'dX', 'ddX', 'theta', 'dtheta', 'ddtheta', 'torque'};

nb_repeats = 25;
mean_perfs = zeros(length(types),4);

for j = 1:length(types)
    train_perf = zeros(nb_repeats,2);
    test_perf = zeros(nb_repeats,2);
    train_fold = zeros(nb_repeats,2);
    test_fold = zeros(nb_repeats,1);
    for i = 1:nb_repeats
        train_folds = round(9*rand(1,2)+1);
        test_folds = round(9*rand()+1);
        while length(unique([train_folds test_folds])) ~= length([train_folds test_folds])
            train_folds = round(9*rand(1,2)+1);
            test_folds = round(9*rand()+1);
        end
        train_fold(i,:) = train_folds;
        test_fold(i,:) = test_folds;
    
        set_train(train_folds);
        set_test(test_folds);
        set_type(types{j});
        
        train;
        
        train_perf(i,:) = parms.net.fvaf_train;
        test_perf(i,:) = parms.net.fvaf_test;
    end
    mean_perfs(j,:) = round([mean(train_perf,1) mean(test_perf,1)],3);
end
% Computing statistics
p_global = anova1(mean_perfs',[],'off'); % ANOVA with each type as a group. Big value == insignificant difference
ps = zeros(length(types));
for i = 1:length(types)
    for j = 1:length(types)
        ps(i,j) = anova1([mean_perfs(i,:); mean_perfs(j,:)]',[],'off');
    end
end
fig0 = figure;
imagesc(ps);
xticklabels(types);
yticklabels(types);
colorbar;
title(['p Values (global p = ' num2str(p_global) ', ' num2str(nb_repeats) ' repeats)']);
% saveas(fig,'Plots/task5_ps.png');

% Showing results
figure;
fig = bar(mean_perfs);

fig(1).FaceColor = [1 0 0];
fig(2).FaceColor = [.5 0 0];
fig(3).FaceColor = [0 0 1];
fig(4).FaceColor = [0 0 .5];

xlabel('Data type')
ylabel('FVAF')
title(['Performance as a function of data type (' num2str(nb_repeats) ' repeats)'])
legend({'Train - 1','Train - 2', 'Test - 1', 'Test - 2'},'Location','best')
xticklabels(types)
% saveas(fig,'Plots/task5_bars.png')
% close all;

%% ----- TASK 6 -----
disp('----- TASK 6 -----')

step_size = 2;
PCA_out_sizes = 1:step_size:960;
nb_repeats = 5;
mean_perfs = zeros(length(PCA_out_sizes),5);


for j = 1:length(PCA_out_sizes)
    train_perf = zeros(nb_repeats,2);
    test_perf = zeros(nb_repeats,2);
%     train_fold = zeros(nb_repeats,2);
%     test_fold = zeros(nb_repeats,1);
    for i = 1:nb_repeats
        train_folds = round(9*rand(1,2)+1);
        test_folds = round(9*rand()+1);
        while length(unique([train_folds test_folds])) ~= length([train_folds test_folds])
            train_folds = round(9*rand(1,2)+1);
            test_folds = round(9*rand()+1);
        end
%         train_fold(i,:) = train_folds;
%         test_fold(i,:) = test_folds;
    
        set_train(train_folds);
        set_test(test_folds);
        set_type('torque');
        set_pca(PCA_out_sizes(j));

        
        train;
        
        train_perf(i,:) = parms.net.fvaf_train;
        test_perf(i,:) = parms.net.fvaf_test;
    end
    mean_perfs(j,:) = round([PCA_out_sizes(j) mean(train_perf,1) mean(test_perf,1)],3);
end

hold on;
fig = plot(mean_perfs(:,1),mean_perfs(:,2),'color',[1 0 0]); % 'Train - Shoulder'
plot(mean_perfs(:,1),mean_perfs(:,3),'color',[0.5 0 0]); % 'Train - Elbow'
plot(mean_perfs(:,1),mean_perfs(:,4),'color',[0 0 1]); % 'Test - Shoulder'
plot(mean_perfs(:,1),mean_perfs(:,5),'color',[0 0 0.5]); % 'Test - Elbow'
xlabel('Number of PCA output dimensions');
ylabel('FVAF');
title(['Performance as a function of dimentionnality reduction (' num2str(nb_repeats) ' repeats, ' num2str(step_size) ' as step size)']);
legend({'Train - Shoulder','Train - Elbow', 'Test - Shoulder', 'Test - Elbow'},'Location','best');
saveas(fig,['Plots/task6_PCA.png']);
% close all;


%% ----- TASK 7 -----
disp('----- TASK 7 -----')

PCA_comp = 0;
nb_repeats = 10;
means = zeros(4);

for j = 0:1
    train_fold = zeros(nb_repeats,2);
    test_fold = zeros(nb_repeats,1);
    for i = 1:nb_repeats
        train_folds = round(9*rand(1,2)+1);
        test_folds = round(9*rand()+1);
        while length(unique([train_folds test_folds])) ~= length([train_folds test_folds])
            train_folds = round(9*rand(1,2)+1);
            test_folds = round(9*rand()+1);
        end
        train_fold(i,:) = train_folds;
        test_fold(i,:) = test_folds;
    
        set_train(train_folds);
        set_test(test_folds);
        set_type('torque');
        set_proprioceptive(j);
        set_pca(PCA_comp);
      
        train;
        
        train_perf(i,:) = parms.net.fvaf_train;
        test_perf(i,:) = parms.net.fvaf_test;
    end
    means(j+1,:) = round([mean(train_perf,1) mean(test_perf,1)],3);
end
for j = 0:1
    train_fold = zeros(nb_repeats,2);
    test_fold = zeros(nb_repeats,1);
    for i = 1:nb_repeats
        train_folds = round(9*rand(1,2)+1);
        test_folds = round(9*rand()+1);
        while length(unique([train_folds test_folds])) ~= length([train_folds test_folds])
            train_folds = round(9*rand(1,2)+1);
            test_folds = round(9*rand()+1);
        end
        train_fold(i,:) = train_folds;
        test_fold(i,:) = test_folds;
    
        set_train(train_folds);
        set_test(test_folds);
        set_type('ddtheta');
        set_proprioceptive(j);
        set_pca(PCA_comp);
      
        train;
        
        train_perf(i,:) = parms.net.fvaf_train;
        test_perf(i,:) = parms.net.fvaf_test;
    end
    means(j+3,:) = round([mean(train_perf,1) mean(test_perf,1)],3);
end

fig = bar(means);

fig(1).FaceColor = [1 0 0];
fig(2).FaceColor = [.5 0 0];
fig(3).FaceColor = [0 0 1];
fig(4).FaceColor = [0 0 .5];

xlabel('Data type and proprioception')
ylabel('FVAF')
title(['Effect of proprioception on performance (' num2str(nb_repeats) ' repeats)'])
legend({'Train - Shoulder','Train - Elbow', 'Test - Shoulder', 'Test - Elbow'},'Location','best')
xticklabels({'Torque - control', 'Torque - proprioception', 'ddtheta - control', 'ddtheta - proprioception'})

% Checking statistical significance
p_torque = anova1(means(1:2,:)',[],"off");
p_theta = anova1(means(3:4,:)',[],"off");

%% ----- TASK 8 -----
disp('----- TASK 8 -----')

delays = 1:20;
nb_repeats = 10;
types = {'X', 'dX', 'ddX', 'theta', 'dtheta', 'ddtheta', 'torque'};

set_pca(0);

for k = 1:length(types)
    mean_perfs = zeros(length(delays),5);
    
    set_type(types{k});

    for j = 1:length(delays)
        train_perf = zeros(nb_repeats,2);
        test_perf = zeros(nb_repeats,2);
        for i = 1:nb_repeats
            train_folds = round(9*rand(1,2)+1);
            test_folds = round(9*rand()+1);
            while length(unique([train_folds test_folds])) ~= length([train_folds test_folds])
                train_folds = round(9*rand(1,2)+1);
                test_folds = round(9*rand()+1);
            end
        
            set_train(train_folds);
            set_test(test_folds);
            set_delays(1:delays(j));
    
            train;
            
            train_perf(i,:) = parms.net.fvaf_train;
            test_perf(i,:) = parms.net.fvaf_test;
        end
        mean_perfs(j,:) = round([delays(j) mean(train_perf,1) mean(test_perf,1)],3);
    end
    
    
    fig = figure;
    hold on;
    plot(mean_perfs(:,1),mean_perfs(:,2),'color',[1 0 0]); % 'Train - Shoulder'
    plot(mean_perfs(:,1),mean_perfs(:,3),'color',[0.5 0 0]); % 'Train - Elbow'
    plot(mean_perfs(:,1),mean_perfs(:,4),'color',[0 0 1]); % 'Test - Shoulder'
    plot(mean_perfs(:,1),mean_perfs(:,5),'color',[0 0 0.5]); % 'Test - Elbow'
    xlabel('Delay [ms]');
    ylabel('FVAF');
    title([types{k} ' performance as a function of delays (' num2str(nb_repeats) ' repeats)']);
    legend({'Train - Shoulder','Train - Elbow', 'Test - Shoulder', 'Test - Elbow'},'Location','best');
    xticks([0 5 10 15 20]);
    xticklabels({'0:50', '0:250', '0:500', '0:750', '0:1000'});
    saveas(fig,['Plots/task8_delays_' types{k} '.png']);
end

% close all;