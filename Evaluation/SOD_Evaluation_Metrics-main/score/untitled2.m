% 方法名列表
method_names = {'CSRNet', 'IRFS','LSNet', 'MIA', 'MIDD','MMNet','Ours'}; % 跟据你实际的方法名进行替换
dataset_name = 'VT1000'
colors = {[0 0.4470 0.7410], [0.8500 0.3250 0.0980], [0.9290 0.6940 0.1250], [0.4940 0.1840 0.5560], [0.4660 0.6740 0.1880], [0.3010 0.7450 0.9330], [1 0 0]};
figure; hold on;
for i = 1:length(method_names)
    % 打开文件
    file_path = sprintf('D:\\pythonproject\\SOD_Evaluation_Metrics-main\\score\\curve_cache\\%s\\%s\\pr.txt',dataset_name ,method_names{i});
    fileID = fopen(file_path,'r');

    % 读取数据
    data = textscan(fileID, '%f%f', 'Delimiter', '\t');

    % 关闭文件
    fclose(fileID);

    % 提取数据
    precision = data{1};
    recall = data{2};

    % 绘制PR曲线，为每个方法指定一个标签
    plot(recall, precision, 'DisplayName', method_names{i}, 'LineWidth', 2, 'Color', colors{i});
end

% 添加图例、标题和标签
legend();
xlabel('Recall');
ylabel('Precision');
title(dataset_name);
grid on;
hold off;