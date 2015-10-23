%problem 4

clear();

addpath(genpath('DeepLearnToolbox-master'));

%read in the inputs
data = dlmread('yeast_input.data');


%read in the outputs
fileID = fopen('yeast_output.data');
C = textscan(fileID,'%s');
fclose(fileID);

%this is just for reformating
textOutputs = C{1, 1};

% now we need to encode the outputs
outputs = zeros(size(textOutputs, 1), 10);
for k = 1:size(textOutputs,1)
   if(strcmp(textOutputs{k}, 'CYT') == 1)
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
trainSplit = randomize(1:floor(1484*.65));
testSplit = randomize(floor(1484*.65)+1:1484);

trainInput = data(trainSplit,:);
trainOuput = outputs(trainSplit,:);
testInput = data(testSplit,:);
testOuput = outputs(testSplit,:);

totalErrors = zeros(3,4); %to store all the total errors

for i = 1:4  %itterating throught the networks fo one hidden layer
    errors = zeros(100,1); %stores the errors at each itteration
    for j = 1:100
        % setting up the neural network
        network1 = nnsetup([8 3*i 10]);
        network1.activation_function = 'perceptron';
        network1.learningRate = .05; %this was suggested by the toolbox creator
        opts.numepochs = 1; %we'll do one runthrough for the whole data set
        opts.batchsize = 1; %were inputing one sample at a time
        opts.plot = 0;
        network1.testing = 0;
        network1.plotting = 0;
        network1.plotting2 = 0;
        
        [network1, ~] = nntrain2(network1, trainInput, trainOuput, opts);
        network1.testing = 1;
        network1.plotting = 0;
        [network1, totalError] = nntrain2(network1, testInput, testOuput, opts);
        errors(j) = totalError;
    end
    totalErrors(1,i) = mean(errors); %store the errors
end

for i = 1:4  %itterating throught the networks for two hidden layer
    errors = zeros(100,1); %stores the errors at each itteration
    for j = 1:100
        % setting up the neural network
        network1 = nnsetup2([8 3*i 3*i 10]);
        network1.activation_function = 'perceptron';
        network1.learningRate = .05; %this was suggested by the toolbox creator
        opts.numepochs = 1; %we'll do one runthrough for the whole data set
        opts.batchsize = 1; %were inputing one sample at a time
        opts.plot = 0;
        network1.testing = 0;
        network1.plotting = 0;
        network1.plotting2 = 0;
        
        [network1, ~] = nntrain2(network1, trainInput, trainOuput, opts);
        network1.testing = 1;
        network1.plotting = 0;
        [network1, totalError] = nntrain2(network1, testInput, testOuput, opts);
        errors(j) = totalError;
    end
    totalErrors(2,i) = mean(errors); %store the errors
end

for i = 1:4  %itterating throught the networks fo one hidden layer
    errors = zeros(100,1); %stores the errors at each itteration
    for j = 1:100
        % setting up the neural network
        network1 = nnsetup2([8 3*i 3*i 3*i 10]);
        network1.activation_function = 'perceptron';
        network1.learningRate = .05; %this was suggested by the toolbox creator
        opts.numepochs = 1; %we'll do one runthrough for the whole data set
        opts.batchsize = 1; %were inputing one sample at a time
        opts.plot = 0;
        network1.testing = 0;
        network1.plotting = 0;
        network1.plotting2 = 0;
        
        [network1, ~] = nntrain2(network1, trainInput, trainOuput, opts);
        network1.testing = 1;
        network1.plotting = 0;
        [network1, totalError] = nntrain2(network1, testInput, testOuput, opts);
        errors(j) = totalError;
    end
    totalErrors(3,i) = mean(errors); %store the errors
end

