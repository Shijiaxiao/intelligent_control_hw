% Created By Jiaxiao Shi on 2024/06/16. All rights reserved.

clear;
clc;

% 创建 Mamdani 型模糊控制系统
fis = mamfis('Name', 'WashingMachine');

% 添加输入变量 '污泥含量' 和 '油脂含量'
fis = addInput(fis, [0 100], 'Name', '污泥含量');
fis = addInput(fis, [0 100], 'Name', '油脂含量');

% 定义 '污泥含量' 的模糊集合: 低 (Low), 中 (Medium), 高 (High)
fis = addMF(fis, '污泥含量', 'trimf', [0 0 50], 'Name', 'Low');
fis = addMF(fis, '污泥含量', 'trimf', [0 50 100], 'Name', 'Medium');
fis = addMF(fis, '污泥含量', 'trimf', [50 100 100], 'Name', 'High');

% 定义 '油脂含量' 的模糊集合: 低 (Low), 中 (Medium), 高 (High)
fis = addMF(fis, '油脂含量', 'trimf', [0 0 50], 'Name', 'Low');
fis = addMF(fis, '油脂含量', 'trimf', [0 50 100], 'Name', 'Medium');
fis = addMF(fis, '油脂含量', 'trimf', [50 100 100], 'Name', 'High');

% 添加输出变量 '洗涤时间'
fis = addOutput(fis, [0 60], 'Name', '洗涤时间');

% 定义 '洗涤时间' 的模糊集合: 短 (Short), 中 (Medium), 长 (Long)
fis = addMF(fis, '洗涤时间', 'trimf', [0 0 30], 'Name', 'Short');
fis = addMF(fis, '洗涤时间', 'trimf', [0 30 60], 'Name', 'Medium');
fis = addMF(fis, '洗涤时间', 'trimf', [30 60 60], 'Name', 'Long');

% 添加模糊规则
ruleList = [
    "污泥含量==Low & 油脂含量==Low => 洗涤时间=Short (1)"
    "污泥含量==Medium | 油脂含量==Medium => 洗涤时间=Medium (1)"
    "污泥含量==High | 油脂含量==High => 洗涤时间=Long (1)"
];
fis = addRule(fis, ruleList);

% 设置模糊逻辑选项
fis.DefuzzificationMethod = 'mom'; % 中间最大值
fis.AndMethod = 'min';             % AND 操作使用 min
fis.OrMethod = 'max';              % OR 操作使用 max
fis.ImplicationMethod = 'min';     % 蕴涵操作使用 min
fis.AggregationMethod = 'max';     % 聚合操作使用 max

% 显示模糊控制系统结构
figure
plotfis(fis)

% 使用 Fuzzy Logic Designer 应用程序查看和编辑模糊控制器
% 使用 'fuzzyLogicDesigner' 打开设计器应用
fuzzyLogicDesigner(fis)

% 测试模糊控制器，输入 x=80, y=60
input = [80 60];
output = evalfis(fis, input);
disp(['输入变量 x=80, y=60 时，输出变量 t = ', num2str(output)])


% 观测不同清晰化方法对输出量的影响
% 清晰化方法列表
defuzzMethods = {'centroid', 'bisector', 'mom', 'lom', 'som'};
results = zeros(length(defuzzMethods), 1);

% 输入变量
input = [70 80];

% 记录不同清晰化方法下的输出结果
for i = 1:length(defuzzMethods)
    fis.DefuzzificationMethod = defuzzMethods{i};
    results(i) = evalfis(fis, input);
end

% 显示不同清晰化方法下的输出结果
T = table(defuzzMethods', results, 'VariableNames', {'DefuzzificationMethod', 'Output'});
disp(T);
