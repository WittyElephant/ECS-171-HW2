function [nn, incorrect, output, trainingError] = nnff(nn, x, y)
%NNFF performs a feedforward pass
% nn = nnff(nn, x, y) returns an neural network structure with updated
% layer activations, error and loss (nn.a, nn.e and nn.L)

    n = nn.n;
    m = size(x, 1);
    
    x = [ones(m,1) x];
    nn.a{1} = x;
    nn.A{1} = nn.a{1}; %the actual activation function


    %feedforward pass
    for i = 2 : n-1
        switch nn.activation_function 
            case 'sigm'
                % Calculate the unit's outputs (including the bias term)
                nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}');

            case 'tanh_opt'
                nn.a{i} = tanh_opt(nn.a{i - 1} * nn.W{i - 1}');
            case 'perceptron' %this is the part I added
                nn.a{i} = sigm(nn.a{i - 1} * nn.W{i - 1}'); %W^tx
                % I really should of just used the sigmoid
                % but I misunderstood the problem at the begining and I
                % don't want to go back and change things (might get bugs)
        end
        
        %dropout
        if(nn.dropoutFraction > 0)
            display('error') %debugger
            if(nn.testing)
                nn.a{i} = nn.a{i}.*(1 - nn.dropoutFraction);
            else
                nn.dropOutMask{i} = (rand(size(nn.a{i}))>nn.dropoutFraction);
                nn.a{i} = nn.a{i}.*nn.dropOutMask{i};
            end
        end
        
        %calculate running exponential activations for use with sparsity
        if(nn.nonSparsityPenalty>0)
            nn.p{i} = 0.99 * nn.p{i} + 0.01 * mean(nn.a{i}, 1);
        end
        
        %Add the bias term
        nn.a{i} = [ones(m,1) nn.a{i}];
            
     end

    switch nn.output 
        case 'sigm'

            nn.a{n} = sigm(nn.a{n - 1} * nn.W{n - 1}');
        case 'linear'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
        case 'softmax'
            nn.a{n} = nn.a{n - 1} * nn.W{n - 1}';
            nn.a{n} = exp(bsxfun(@minus, nn.a{n}, max(nn.a{n},[],2)));
            nn.a{n} = bsxfun(@rdivide, nn.a{n}, sum(nn.a{n}, 2)); 
    end

    %error and loss
    trainingError = 1;                  %the total error for training
    nn.e = y - nn.a{n};
    incorrect = zeros(1,10);            %this is for errors for each clasification
    for i = 1:10
        if(y(i) == 1)                       %this is our sample
            if(max(nn.a{n}) == nn.a{n}(i))  %and we guess right
                trainingError = 0;          %there's no error
            end
        
        else                                %otherwise if this isn't our sample
            if(max(nn.a{n}) == nn.a{n}(i))  %and we still guessed this
                    incorrect(i)= 1;        %then we errored for that sample
            end
        end
        
    end
     
   
    output = nn.a{n}(:);

    switch nn.output
        case {'sigm', 'linear'}
            nn.L = 1/2 * sum(sum(nn.e .^ 2)) / m; 
        case 'softmax'
            nn.L = -sum(sum(y .* log(nn.a{n}))) / m;
    end
end
