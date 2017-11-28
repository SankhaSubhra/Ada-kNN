function [x] = standardised(x1)

% Data standardization.

xMean=mean(x1);
xStd=std(x1);
x=zeros(size(x1));

zerosIdx=(xStd==0);
temp1=x1(:, ~zerosIdx)-repmat(xMean(:, ~zerosIdx), size(x1(:, ~zerosIdx), 1), 1);
temp2=repmat(xStd(:, ~zerosIdx), size(x1(:, ~zerosIdx), 1), 1);
x(:, ~zerosIdx)=temp1./temp2;
x(:, zerosIdx)=0;


end

