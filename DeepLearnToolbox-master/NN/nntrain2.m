function [nn, totalError]  = nntrain2(nn, train_x, train_y, opts, val_x, val_y)
%NNTRAIN trains a neural net
% [nn, L] = nnff(nn, x, y, opts) trains the neural network nn with input x and
% output y for opts.numepochs epochs, with minibatches of size
% opts.batchsize. Returns a neural network nn with updated activations,
% errors, weights and biases, (nn.a, nn.e, nn.W, nn.b) and L

%this was specifically fro problem 4 as graphing the weights was a problem
%for neural networks of different sizes





assert(isfloat(train_x), 'train_x must be a float');
assert(nargin == 4 || nargin == 6,'number ofinput arguments must be 4 or 6')

loss.train.e               = [];
loss.train.e_frac          = [];
loss.val.e                 = [];
loss.val.e_frac            = [];
opts.validation = 0;
if nargin == 6
    opts.validation = 1;
end

fhandle = [];
if isfield(opts,'plot') && opts.plot == 1
    fhandle = figure();
end

m = size(train_x, 1);

batchsize = opts.batchsize;
numepochs = opts.numepochs;
numbatches = m / batchsize;

%the data stores for graphing purposes
outputs = zeros(numbatches,10);
sampleErrors = zeros(numbatches, 10);
numError = zeros(1, 10);
trainingError = zeros(numbatches, 1);
errorCount = 0;
epochErrors = zeros(numepochs, 1);
a12 = zeros(numbatches, 9); % weights for node 1 layer 2
a22 = zeros(numbatches, 9);
a32 = zeros(numbatches, 9);
a13 = zeros(numbatches, 4);
a23 = zeros(numbatches, 4);
a33 = zeros(numbatches, 4);
a43 = zeros(numbatches, 4);
a53 = zeros(numbatches, 4);
a63 = zeros(numbatches, 4);
a73 = zeros(numbatches, 4);
a83 = zeros(numbatches, 4);
a93 = zeros(numbatches, 4);
a103 = zeros(numbatches, 4);

testing = nn.testing;

assert(rem(numbatches, 1) == 0, 'numbatches must be a integer');

L = zeros(numepochs*numbatches,1);
n = 1;
for i = 1 : numepochs
    tic;
    
    sampleErrors = zeros(numbatches, 10);
    numError = zeros(1, 10);
    trainingError = zeros(numbatches, 1);
    errorCount = 0;
    
    kk = randperm(m);
    for l = 1 : numbatches
        batch_x = train_x(kk((l - 1) * batchsize + 1 : l * batchsize), :);
        
        %Add noise to input (for use in denoising autoencoder)
        if(nn.inputZeroMaskedFraction ~= 0)
            batch_x = batch_x.*(rand(size(batch_x))>nn.inputZeroMaskedFraction);
        end
        
        batch_y = train_y(kk((l - 1) * batchsize + 1 : l * batchsize), :);

        
        [nn, incorrect, output, sampleCorrect] = nnff(nn, batch_x, batch_y);

        outputs(l,:) = output';    %storing stuf for graphing later
        numError = numError + incorrect; %numError = numerator errors for each sample
        
            
         errorCount = errorCount + sampleCorrect;
         trainingError(l) = errorCount/l;

        sampleErrors(l,:) = (numError/l);
        if(testing ==0) %if we are testing then we dont want to modify any of the weights
            nn = nnbp(nn);
            nn = nnapplygrads(nn);
            a12(l,:) = nn.W{1}(1,:); % weights for node 1 layer 2
            a22(l,:) = nn.W{1}(2,:);
            a32(l,:) = nn.W{1}(3,:);
%             a13(l,:) = nn.W{2}(1,:);
%             a23(l,:) = nn.W{2}(2,:);
%             a33(l,:) = nn.W{2}(3,:);
%             a43(l,:) = nn.W{2}(4,:);
%             a53(l,:) = nn.W{2}(5,:);
%             a63(l,:) = nn.W{2}(6,:);
%             a73(l,:) = nn.W{2}(7,:);
%             a83(l,:) = nn.W{2}(8,:);
%             a93(l,:) = nn.W{2}(9,:);
%             a103(l,:) = nn.W{2}(10,:);
        end
        
        L(n) = nn.L;
        
        n = n + 1;
    end
    
    t = toc;

    if opts.validation == 1
        loss = nneval(nn, loss, train_x, train_y, val_x, val_y);
        str_perf = sprintf('; Full-batch train mse = %f, val mse = %f', loss.train.e(end), loss.val.e(end));
    else
        loss = nneval(nn, loss, train_x, train_y);
        str_perf = sprintf('; Full-batch train err = %f', loss.train.e(end));
    end
    if ishandle(fhandle)
        nnupdatefigures(nn, fhandle, loss, opts, i);
    end
    

    totalError = trainingError(numbatches);
    epochErrors(i) = totalError;
    nn.learningRate = nn.learningRate * nn.scaling_learningRate;
end


outputs= outputs';
if(nn.plotting ==1)

    figure
    plot(epochErrors);
    title('Error Progression of Error for Every Epoch')
    xlabel('Sample Number')
    ylabel('Error Percentage')
    legend('Total Error');
    axis([0 inf 0 1]);
    
    
end

if(nn.plotting2 ==1)
        figure
    plot(trainingError);
    title('Error Progression of Total Error for Dataset Run')
    xlabel('Sample Number')
    ylabel('Error Percentage')
    legend('Total Error');
    axis([0 inf 0 1]);
    
    figure
    for i = 1:5
        plot(outputs(i,:));
        hold on
    end
    title('Outputs of CTY, NUC, MIT, ME3, and ME2')
    xlabel('Sample Number')
    ylabel('Output value')
    legend('CTY', 'NUC', 'MIT', 'ME3','ME2');
    axis([0 inf 0 1]);
    hold off
    
    figure
    for i = 6:10
        plot(outputs(i,:));
        hold on
    end
    title('Outputs of ME2, ME1, EXC, POX, and ERL')
    xlabel('Sample Number')
    ylabel('Output value')
    legend('ME2', 'ME1', 'EXC', 'POX', 'ERL');
    axis([0 inf 0 1]);
    hold off
    
    figure
    for i = 1:5
        plot(sampleErrors(:,i))
        hold on
    end
    title('Error Progression of CTY, NUC, MIT, ME3, and ME2')
    xlabel('Sample Number')
    ylabel('Error Percentage')
    legend('CTY', 'NUC', 'MIT', 'ME3','ME2');
    axis([0 inf 0 1]);
    hold off
    
    figure
    for i = 6:10
        plot(sampleErrors(:,i))
        hold on
    end
    title('Error Progression of ME2, ME1, EXC, POX, and ERL')
    xlabel('Sample Number')
    ylabel('Error Percentage')
    legend('ME2', 'ME1', 'EXC', 'POX', 'ERL');
    axis([0 inf 0 1]);
    hold off

    figure
    for i = 1:9
        plot(a12(:,i))
        hold on
    end
    title('Weight Change Progression for Node 1 of layer2 (Hidden Layer)')
    xlabel('Sample Number')
    ylabel('Weight Value')
    legend('W0', 'W1', 'W2', 'W3', 'W4', 'W5','W6', 'W7', 'W8');
    hold off
    
    figure
    for i = 1:9
        plot(a22(:,i))
        hold on
    end
    title('Weight Change Progression for Node 2 of layer2 (Hidden Layer)')
    xlabel('Sample Number')
    ylabel('Weight Value')
    legend('W0', 'W1', 'W2', 'W3', 'W4', 'W5','W6', 'W7', 'W8');
    hold off
    
    figure
    for i = 1:9
        plot(a32(:,i))
        hold on
    end
    title('Weight Change Progression for Node 3 of layer2 (Hidden Layer)')
    xlabel('Sample Number')
    ylabel('Weight Value')
    legend('W0', 'W1', 'W2', 'W3', 'W4', 'W5','W6', 'W7', 'W8');
    hold off
    
%     figure
%     for i = 1:4
%         plot(a13(:,i))
%         hold on
%     end
%     title('Weight Change Progression for Node 1 of layer3 (output Layer)')
%     xlabel('Sample Number')
%     ylabel('Weight Value')
%     legend('W0', 'W1', 'W2', 'W3');
%     hold off
%     
%     figure
%     for i = 1:4
%         plot(a23(:,i))
%         hold on
%     end
%     title('Weight Change Progression for Node 2 of layer3 (output Layer)')
%     xlabel('Sample Number')
%     ylabel('Weight Value')
%     legend('W0', 'W1', 'W2', 'W3');
%     hold off
%     
%     figure
%     for i = 1:4
%         plot(a33(:,i))
%         hold on
%     end
%     title('Weight Change Progression for Node 3 of layer3 (output Layer)')
%     xlabel('Sample Number')
%     ylabel('Weight Value')
%     legend('W0', 'W1', 'W2', 'W3');
%     hold off
%     
%     figure
%     for i = 1:4
%         plot(a43(:,i))
%         hold on
%     end
%     title('Weight Change Progression for Node 4 of layer3 (output Layer)')
%     xlabel('Sample Number')
%     ylabel('Weight Value')
%     legend('W0', 'W1', 'W2', 'W3');
%     hold off
%     
%     figure
%     for i = 1:4
%         plot(a53(:,i))
%         hold on
%     end
%     title('Weight Change Progression for Node 5 of layer3 (output Layer)')
%     xlabel('Sample Number')
%     ylabel('Weight Value')
%     legend('W0', 'W1', 'W2', 'W3');
%     hold off
%     
%     figure
%     for i = 1:4
%         plot(a63(:,i))
%         hold on
%     end
%     title('Weight Change Progression for Node 6 of layer3 (output Layer)')
%     xlabel('Sample Number')
%     ylabel('Weight Value')
%     legend('W0', 'W1', 'W2', 'W3');
%     hold off
%     
%     figure
%     for i = 1:4
%         plot(a73(:,i))
%         hold on
%     end
%     title('Weight Change Progression for Node 7 of layer3 (output Layer)')
%     xlabel('Sample Number')
%     ylabel('Weight Value')
%     legend('W0', 'W1', 'W2', 'W3');
%     hold off
%     
%     figure
%     for i = 1:4
%         plot(a83(:,i))
%         hold on
%     end
%     title('Weight Change Progression for Node 8 of layer3 (output Layer)')
%     xlabel('Sample Number')
%     ylabel('Weight Value')
%     legend('W0', 'W1', 'W2', 'W3');
%     hold off
%     
%     figure
%     for i = 1:4
%         plot(a93(:,i))
%         hold on
%     end
%     title('Weight Change Progression for Node 9 of layer3 (output Layer)')
%     xlabel('Sample Number')
%     ylabel('Weight Value')
%     legend('W0', 'W1', 'W2', 'W3');
%     hold off
%     
%     figure
%     for i = 1:4
%         plot(a103(:,i))
%         hold on
%     end
%     title('Weight Change Progression for Node 10 of layer3 (output Layer)')
%     xlabel('Sample Number')
%     ylabel('Weight Value')
%     legend('W0', 'W1', 'W2', 'W3');
%     hold off
end

end

