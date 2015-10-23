%problem 1

clear();

addpath(genpath('DeepLearnToolbox-master'));

%read in the inputs
data = dlmread('nn_tester_input.data');


%read in the outputs
fileID = fopen('nn_tester_output.data');
C = textscan(fileID,'%s');
fclose(fileID);

%this is just for reformating
textOutputs = C{1, 1};

% now we need to encode the outputs
outputs = zeros(size(textOutputs, 1), 10);
for k = 1:size(textOutputs,1)
   if(strcmp(textOutputs{k}, 'CTY') == 1)
       outputs(k, 1) = 1;
   elseif(strcmp(textOutputs{k}, 'NUC') == 1)
       outputs(k, 2) = 1;
   elseif(strcmp(textOutputs{k}, 'MIT') == 1)
       outputs(k, 3) = 1;
   elseif(strcmp(textOutputs{k}, 'ME3') == 1)
       outputs(k, 4) = 1;
   elseif(strcmp(textOutputs{k}, 'ME2') == 1)
       outputs(k, 5) = 1;
   elseif(strcmp(textOutputs{k}, 'ME1') == 1)
       outputs(k, 6) = 1;
   elseif(strcmp(textOutputs{k}, 'EXC') == 1)
       outputs(k, 7) = 1;
   elseif(strcmp(textOutputs{k}, 'VAC') == 1)
       outputs(k, 8) = 1;
   elseif(strcmp(textOutputs{k}, 'POX') == 1)
       outputs(k, 9) = 1;
   else % must be ERL
       outputs(k, 10) = 1;
   end
end

%now we will randomly split the data sets
randomize = randperm(size(outputs,1)); %randomly permutated an array from 1 to 1484
trainSplit = randomize(1:floor(100*.65));
testSplit = randomize((floor(100*.65)+1):100);

trainInput = data(trainSplit,:);
trainOuput = outputs(trainSplit,:);
testInput = data(testSplit,:);
testOuput = outputs(testSplit,:);
%debugger

errors = zeros(100,1);
for i = 1:100
    % setting up the neural network
    network = nnsetup([8 3 3 10]);
    network.activation_function = 'perceptron';
    network.learningRate = .7; %this was suggested by the toolbox creator
    opts.numepochs = 1; %we'll do one runthrough for the whole data set
    opts.batchsize = 1; %were inputing one sample at a time
    opts.plot = 0;
    network.testing = 0;
    network.plotting = 0;
    network.plotting2 = 0;
    
    [network, ~] = nntrain2(network, trainInput, trainOuput, opts);
    network.testing = 1;
    network.plotting = 0;
    network.plotting2 = 0;
    [network, totalError] = nntrain2(network, testInput, testOuput, opts);
    errors(i) = totalError;
end
     disp(mean(errors));
errors = zeros(100,1);
for i = 1:100
    % setting up the neural network
    network = nnsetup([8 3 10]);
    network.activation_function = 'perceptron';
    network.learningRate = .7; %this was suggested by the toolbox creator
    opts.numepochs = 1; %we'll do one runthrough for the whole data set
    opts.batchsize = 1; %were inputing one sample at a time
    opts.plot = 0;
    network.testing = 0;
    network.plotting = 0;
    network.plotting2 = 0;
    
    [network, ~] = nntrain2(network, trainInput, trainOuput, opts);
    network.testing = 1;
    network.plotting = 0;
    network.plotting2 = 0;
    [network, totalError] = nntrain2(network, testInput, testOuput, opts);
    errors(i) = totalError;
end
     disp(mean(errors));
