function data = loadAndAssemble( path )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
directory = dir(path);
dataDirectories = length(directory) - 3;
%compute to see how many hierarchy trees we have
curData = load([path '/data-' num2str(dataDirectories) '.mat']);
len = length(curData.curData);
numTrees = (1000 * (dataDirectories - 1)) + len;
totalData(numTrees) = HierarchyTree();
for i=1:dataDirectories
    disp(['loading up data file ' num2str(i)]);
    start = (i-1) * 1000 + 1;
    nd = start + 999;
    pathToMatFile = [path '/data-' num2str(i) '.mat'];
    curData = load(pathToMatFile);
    if i == dataDirectories
        totalData(start:end) = curData.curData;
    else
        totalData(start:nd) = curData.curData;
    end
end
disp('assembling neighborhood object...')
pathToMatFile = [path '/object.mat'];
data = load(pathToMatFile, 'newData');
data = data.newData;
data.rootData = totalData; 
end

