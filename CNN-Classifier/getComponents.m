function [trainingIms, trainingLabels, validationIms, validationLabels, testIms, testLabels, validationSize, testSize] = getComponents(distribution, validationSize, testingSize, gatewayBool, fullScaleBool)
%picks out num components of type type.

if ~fullScaleBool
	numRows = size(distribution, 1);
	totalComponents = 0;
	for i=1:numRows
	    totalComponents = totalComponents + distribution{i, 2};
	end
	trainingIms = zeros(128,128,3, totalComponents);
	trainingLabels = zeros(totalComponents,1);
	curInd = 0;

	organic = load('/local/scr/mjcurcio/SEMERU/organic-components.mat');
	organic.labels = num2cell(organic.labels);
	organic.groups = num2cell(organic.groups);
	organic = [organic.trainingImages organic.labels organic.groups];
	vd = organic(1:15000,:);
	testData = organic(15001:30000,:);
	organic = organic(30001:end,:);

	synthetic = load('/local/scr/mjcurcio/SEMERU/synthetic-components.mat');
	synthetic.labels = num2cell(synthetic.labels);
	synthetic.groups = num2cell(synthetic.groups);
	synthetic = [synthetic.trainingImages, synthetic.labels synthetic.groups];

	if gatewayBool
	    position = 3;
	else
	    position = 2;
	end
	%training data 
	for j=1:numRows
	%     curList = organic(strcmp(organic{:,1},distribution{j,1}));
	    temp = organic(:,2);
	    for i=1:length(temp)
		temp{i} = char(code2name(temp{i}, false));
	    end
	    logind = strcmp(distribution{j,1},temp);
	    curList = organic(logind,:);
	    num = distribution{j,2};
	    possible = size(curList, 1);
	    %if there arent enough, we resort to the synthetic components
	    if possible < num
	%         synthetics = synthetic(strcmp(synthetic{:,1},distribution{j,1}));
		temp = synthetic(:,2);
		for i=1:length(temp)
		    temp{i}=char(code2name(temp{i}, false));
		end
		logind = strcmp(temp,distribution{j,1});
		synthetics = synthetic(logind,:);
	    end
	    needed = max(0, num - possible);
	    ind = randperm(possible, min(possible,num));
	    files = curList(ind,:);
	    if needed > 0
		ind = randperm(size(synthetics,1), needed);
		synFiles = synthetics(ind, :);
		files = vertcat(files, synFiles);
	    end
	    
	    %read images and put them in the array
	    for i=1:num
		arr = imread(files{i,1});
		newArr = imresize(arr, [128, 128]);
		trainingIms(:,:,:,curInd + i) = newArr;
		trainingLabels(curInd + i,1) = files{i,position};
	    end
	    curInd = curInd + num;
	end
	%validation data
	tempMat = vd(:,position);
	temp = cell(length(tempMat),1);
	for i=1:length(tempMat)
	    temp{i} = char(code2name(tempMat{i}, gatewayBool));
	end
	%logInd1 = strcmp('Button', temp);
	%logInd2 = strcmp('Switch', temp);
	%logInd3 = strcmp('ImageButton', temp);
	%logInd4 = strcmp('CheckBox', temp);
	%logInd5 = strcmp('Spinner', temp);
	%logInd6 = strcmp('RadioButton', temp);
	logInd = ~strcmp('ignore', temp);
	vd = vd(logInd, :);
	vSize = size(vd, 1);
	if vSize >= validationSize
	    ind = randperm(vSize, validationSize);
	else
	    ind = randperm(vSize, vSize);
	end
	validationMat = vd(ind,:);
	temp = validationMat(:,position);
	validationLabels = zeros(length(temp),1);
	for i=1:length(temp)
	    validationLabels(i) = temp{i};
	end
	validationIms = zeros(128,128,3,min(vSize, validationSize));
	validationSize = zeros(size(validationIms,4), 2);
	for i=1:length(validationLabels)
	    arr = imread(validationMat{i,1});
		validationSize(i, 1) = size(arr,2);
		validationSize(i, 2) = size(arr,1);
	    newArr = imresize(arr, [128 128]);
	    validationIms(:,:,:,i) = newArr;
	end
	%testing data
	tempMat = testData(:,position);
	temp = cell(length(tempMat),1);
	for i=1:length(tempMat)
	    temp{i} = char(code2name(tempMat{i}, gatewayBool));
	end
	%logInd1 = strcmp('Button', temp);
	%logInd2 = strcmp('Switch', temp);
	%logInd3 = strcmp('ImageButton', temp);
	%logInd4 = strcmp('CheckBox', temp);
	%logInd5 = strcmp('Spinner', temp);
	%logInd6 = strcmp('RadioButton', temp);
	logInd = ~strcmp('ignore', temp);
	testData = testData(logInd, :);
	tSize = size(testData, 1);
	if tSize >= testingSize
	    ind = randperm(tSize, testingSize);
	else
	    ind = randperm(tSize, tSize);
	end
	testMat = testData(ind,:);
	temp = testMat(:,position);
	testLabels = zeros(length(temp),1);
	for i=1:length(temp)
	    testLabels(i) = temp{i};
	end
	testIms = zeros(128,128,3,testingSize);
	testSize = zeros(size(testIms, 4), 2);
	for i=1:size(testMat,1)
	    arr = imread(testMat{i,1});
		testSize(i, 1) = size(arr, 2);
		testSize(i, 2) = size(arr, 1);
	    newArr = imresize(arr, [128 128]);
	    testIms(:,:,:,i) = newArr;
	end
else
	dataStruct = load('all-data-3.mat');

	%parse out training data
	fields = fieldnames(dataStruct.trd);
	curCellMat = {};
	for i=1:length(fields)
		curCellMat = vertcat(curCellMat, dataStruct.trd.(fields{i}));
	end
	trainingIms = zeros(128,128,3,size(curCellMat,1), 'uint8');
	for i=1:size(curCellMat,1)
		arr = imread(curCellMat{i,1});
		arr = imresize(arr, [128 128]);
		trainingIms(:,:,:,i) = arr;
	end
	trainingLabels = cell2mat(curCellMat(:,2));
	
	randNdx = randperm(length(trainingLabels));
	trainingLabels = trainingLabels(randNdx);
	trainingIms = trainingIms(:,:,:,randNdx);
	
	%parse out validation data
	fields = fieldnames(dataStruct.vd);
	curCellMat = {};
	for i=1:length(fields)
		curCellMat = vertcat(curCellMat, dataStruct.vd.(fields{i}));
	end
	validationIms = zeros(128,128,3,size(curCellMat, 1), 'uint8');
	validationSize = zeros(size(curCellMat, 1), 2);
	for i=1:size(curCellMat, 1)
		arr = imread(curCellMat{i,1});
		validationSize(i,1) = size(arr, 2);
		validationSize(i,2) = size(arr, 1);
		arr = imresize(arr, [128 128]);
		validationIms(:,:,:,i) = arr;
	end
	validationLabels = cell2mat(curCellMat(:,2));
	
	randNdx = randperm(length(validationLabels));
	validationLabels = validationLabels(randNdx);
	validationIms = validationIms(:,:,:,randNdx);
	validationSize = validationSize(randNdx, :);

	%parse out testing data
	fields = fieldnames(dataStruct.ted);
	curCellMat = {};
	for i=1:length(fields)
		curCellMat = vertcat(curCellMat, dataStruct.ted.(fields{i}));
	end
	testIms = zeros(128, 128, 3, size(curCellMat, 1), 'uint8');
	testSize = zeros(size(curCellMat, 1), 2);
	for i=1:size(curCellMat,1)
		arr = imread(curCellMat{i,1});
		testSize(i,1) = size(arr,2);
		testSize(i,2) = size(arr,1);
		arr = imresize(arr, [128, 128]);
		testIms(:,:,:,i) = arr;
	end
	testLabels = cell2mat(curCellMat(:,2));
	
	randNdx = randperm(length(testLabels));
	testLabels = testLabels(randNdx);
	testIms = testIms(:,:,:,randNdx);
	testSize = testSize(randNdx, :);
end
	

%switch all label vectors to categorical
trainingLabels = categorical(trainingLabels);
validationLabels = categorical(validationLabels);
testLabels = categorical(testLabels);
end

