%% Epoch Iteration Graph

% Epochs are how many times a full dataset is passed forwards and backwards
% through the neural network. Every time a new epoch is passed through the
% neural network, it will update its weights. If too little epochs are used
% to train the model, the result will be an underfitted model. However, if
% too many epochs are used to train the model, the model will be overfitted
% and will end up being accurate as well. The convolutional neural network
% in this project was modeled at many different epoch numbers, and the
% results are tabulated below:

epoch = 1:100;

lossAcc_Data = csvread("epochIterationVals_relu.csv");
relu_loss = lossAcc_Data(:,1);
relu_accuracy = lossAcc_Data(:,2);

lossAcc_Data = csvread("epochIterationVals_softmax.csv");
softmax_loss = lossAcc_Data(:,1);
softmax_accuracy = lossAcc_Data(:,2);

subplot(2, 1, 1)
plot(epoch, [relu_loss, softmax_loss])
legend("ReLU (%)", "Softmax (%)")
title("Comparison of Loss (ReLU vs. Softmax)")

subplot(2, 1, 2)
plot(epoch, [relu_accuracy, softmax_accuracy])
legend("ReLU (%)", "Softmax (%)")
title("Comparison of Accuracy (ReLU vs. Softmax)")
xlabel("Epoch Iteration")

% From the above graph, we see that at larger epochs, Softmax outperforms
% ReLU. We next attempt to determine what epoch value to use.
