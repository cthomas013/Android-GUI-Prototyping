% Script to build the classifier which we will use to 
%  build our object detector
VALIDATION_SIZE = 2000;
TESTING_SIZE = 2000;
%setting this to true will train a gateway net instead of leaf-level.
TRAIN_GATEWAY_NET = false;
TRAIN_FULL_SCALE_NET = true;
distribution = {
%large group
'TextView', 200;
'ImageView', 200;

%button group
'Button', 200;
'Switch', 200;
'ImageButton', 200;
'ToggleButton', 200;
'CheckBox', 200;
'Spinner', 200;
'RadioButton', 200;

%test group
'CheckedTextView', 200;
'EditText', 200;

%bar group
'ProgressBar', 200;
'RatingBar', 200;
'SeekBar', 200;

%number pickers
'NumberPicker', 200;
};
disp('Loading up and formatting data for training...');

%[trainingIms, trainingLabels, validationIms, validationLabels, testIms, testLabels, validationSize, testSize] = getComponents(distribution,VALIDATION_SIZE, TESTING_SIZE, TRAIN_GATEWAY_NET, ...
%																	TRAIN_FULL_SCALE_NET);

%put where checkpoints are to go
NETWORK_CHECKPOINT_LOCATION = '/local/scr/mjcurcio/ReDraw/Android-CNN-Component-Detector/MATLAB-R-CNN/Android-Workspace/NetCheckpoints';
%age
msg = 'Finished formatting data';
disp(msg);
%size of the resized cropped out android views
SCREEN_WIDTH = 128;
SCREEN_HEIGHT = 128;
FILTER_SIZE = 7;

if ~TRAIN_GATEWAY_NET
	comps = cell2mat(distribution(:,2));
	indices = find(comps);
	NUM_IMAGE_CATEGORIES = length(comps(indices));
else
	NUM_IMAGE_CATEGORIES = 5;
end

%input layer
imSize = [SCREEN_HEIGHT SCREEN_WIDTH 3];
inputLayer = imageInputLayer(imSize);

%We want 3 convolutional layers, trying to preserve as much
% image dimension as possible so as to detect important low level features
numFilters = 64;
middleLayers = [
    convolution2dLayer(7, numFilters, 'Padding', 3, 'stride', 2)
    reluLayer()
    convolution2dLayer(5, 128, 'padding', 2, 'stride', 2)
    reluLayer()
    maxPooling2dLayer(3, 'stride', 1)
    convolution2dLayer(3, 96)
    reluLayer()
    maxPooling2dLayer(2)
    dropoutLayer(0.5)
    ];

finalLayers = [
    %since our images are roughly double those found in cnn.py, we
    % double the number of nodes
    fullyConnectedLayer(1024)
    reluLayer()
    dropoutLayer(0.5)
    fullyConnectedLayer(1024)
    reluLayer()
    fullyConnectedLayer(NUM_IMAGE_CATEGORIES)
    softmaxLayer()
    classificationLayer()
    ];

netLayers = [inputLayer; middleLayers; finalLayers];
%initialize weights
layers(2).weights = 0.0001 * randn([FILTER_SIZE, 3, numFilters]);
%these are typical hyperparameters, subject to change
opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 1, ...
    'MiniBatchSize', 256, ...
    'Verbose', true, ...
    'VerboseFrequency', 10, ...
    'CheckpointPath', NETWORK_CHECKPOINT_LOCATION, ...
    'ExecutionEnvironment', 'gpu');
nfiles = length(trainingIms);

disp('training epoch 1...');
deleteLocation = [NETWORK_CHECKPOINT_LOCATION '*.mat'];
%input location of checkpoings + '*.mat'
if length(dir(NETWORK_CHECKPOINT_LOCATION)) > 2
	movefile '/local/scr/mjcurcio/ReDraw/Android-CNN-Component-Detector/MATLAB-R-CNN/Android-Workspace/NetCheckpoints/*.mat'  '/local/scr/mjcurcio/ReDraw/Android-CNN-Component-Detector/MATLAB-R-CNN/Android-Workspace/CheckpointArchive';
end
tic;
classifier = trainNetwork(trainingIms, trainingLabels, netLayers, opts);

disp('taking validation data...');
[mat, per] = testClassifier(classifier, validationIms, validationLabels, TRAIN_GATEWAY_NET);
validation3dMat = cell(size(mat,1),size(mat,2),100);
validation3dMat(:,:,1) = mat;
disp(['validation accuracy ' num2str(per)]);
validationAccuracy = [1, per];

lr = 0.000001;
% validationAccuracy = per;
for i=2:200
	if mod(i,50) == 0
		lr = lr / 10; 
	end
		opts = trainingOptions('sgdm', ...
        'Momentum', 0.9, ...
        'InitialLearnRate', 0.00001, ...
        'L2Regularization', 0.004, ...
        'MaxEpochs', 1, ...
        'MiniBatchSize', 256, ...
        'Verbose', true, ...
        'VerboseFrequency', 10, ...
        'CheckpointPath', NETWORK_CHECKPOINT_LOCATION, ...
        'ExecutionEnvironment', 'gpu');

%    if i >= 50 && i <=74
%        opts = trainingOptions('sgdm', ...
%        'Momentum', 0.9, ...
%        'InitialLearnRate', 0.00001, ...
%        'L2Regularization', 0.004, ...
%        'MaxEpochs', 1, ...
%        'MiniBatchSize', 256, ...
%        'Verbose', true, ...
%        'VerboseFrequency', 10, ...
%        'CheckpointPath', NETWORK_CHECKPOINT_LOCATION, ...
%        'ExecutionEnvironment', 'gpu');
%    elseif i >=75
%        opts = trainingOptions('sgdm', ...
%        'Momentum', 0.9, ...
%        'InitialLearnRate', 0.000001, ...
%        'L2Regularization', 0.004, ...
%        'MaxEpochs', 1, ...
%        'MiniBatchSize', 256, ...
%        'Verbose', true, ...
%        'VerboseFrequency', 10, ...
%        'CheckpointPath', NETWORK_CHECKPOINT_LOCATION, ...
%        'ExecutionEnvironment', 'gpu');
%    end
    
    list = dir(NETWORK_CHECKPOINT_LOCATION);
    checkpoint = list(length(list));
    cp = [checkpoint.folder '/' checkpoint.name];
    load(cp);
    %clear out the directory after loading to prevent grabbing the wrong
    %checkpoint
    %input location of checkpoings + '*.mat'
    movefile '/local/scr/mjcurcio/ReDraw/Android-CNN-Component-Detector/MATLAB-R-CNN/Android-Workspace/NetCheckpoints/*.mat'  '/local/scr/mjcurcio/ReDraw/Android-CNN-Component-Detector/MATLAB-R-CNN/Android-Workspace/CheckpointArchive';
    
    disp(['training epoch' ' ' num2str(i) '...']);
    classifier = trainNetwork(trainingIms, trainingLabels, net.Layers, opts);
    if mod(i, 5)==0
	    disp('taking validation data...');
	    [validationMat, per] = testClassifier(classifier, validationIms, validationLabels, TRAIN_GATEWAY_NET);
	    disp(['validation accuracy ' num2str(per)]);
	    validationAccuracy = [validationAccuracy; i, per];
	    validation3dMat(:,:,i) = validationMat;
    end

	%early stopping functionality
	if size(ValidationAccuracy,1) >= 6
			newMean = mean(ValidationAccuracy(end-2:end,2));
			oldMean = mean(ValidationAccuracy(end-5:end-3, 2));
			if newMean < oldMean
				disp(['overfitting detected, stopping early at iteration' num2str(i)]);
				break;
			end
	end
end
elapsed = toc;
disp('training completed');
disp(['training time: ' num2str(elapsed)]);
    
