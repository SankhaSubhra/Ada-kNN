function [ p, accuracy, gmeans ] = adaKnn2GIHS( trainS,label, testS, labelT, k_max )

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Implementation of Ada-kNN2+GIHS for imbalanced classification problem.
% Written by: Sankha Subhra Mullick.
% Please see the README in GitHub for the corresponding reference. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Input:
% trainS: training dataset containing m number of d-dimensional data points: matrix of size m*d.
% label: class label corresponding to training set, array of size m*1.
% testS: test dataset containing n number of d-dimensional data points: matrix of size n*d.
% labelT: class label corresponding to test set, array of size n*1.
% k_max: set it to ceil(sqrt(m)). See corresponding article for further information.

% Output:
% p: Predicted class labels of the test points.
% accuracy: value of accuracy index.
% gmeans: value of gmeans index.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Ground work.
trainS=standardised(trainS);
testS=standardised(testS);

% Initialization.
[m, ~] =size(trainS);
[n, ~]=size(testS);
k_new=randperm(k_max);
k_med_new=zeros(n, 1);
trainK=cell(m, 1);
nnDist=zeros(m, 1);
p=zeros(n, 1);
classes=unique(label);
classNum=length(classes);
classAcc=zeros(classNum, 1);
k1_max=ceil(nthroot(m, 3));
rng('shuffle');

% GIHS weights calculation.
ideal=ones(1, classNum)./classNum;
actual=hist(label, classes);
actual=actual./m;
zeroActual=(actual==0);
pClass=ones(1, classNum);
change=zeros(1, classNum);
change(~zeroActual)=(ideal(~zeroActual)-actual(~zeroActual))./actual(~zeroActual);
pClass=pClass+change;
k_Alpha=min(10, k_max);

% adaKNN training phase
for i=1:m
    % Initialization.
    l=1;
    flag=0;
    km=[];
    % Distance calculation and sorting.
    distanceM=zeros(1, m-1);
    for f=2:m
        distanceM(f-1)=norm(trainS(1, :)-trainS(f, :));
    end
    [distanceM, distanceI]=sort(distanceM, 2, 'ascend');
    nnDist(i)=distanceM(1);
    % Findindg the successful choices of k.
    for k=1:k_Alpha
        h=kNNImb(label(2:end), k_new(k), distanceI, classes, pClass);
        if h==label(1)
            flag=1;
            km(l)=k_new(k);
            l=l+1;
        end
    end
    
    if(flag~=1)
        for k=11:k_max
            h=kNNImb(label(2:end), k_new(k), distanceI, classes, pClass);
			if h==label(1)
                km(1)=k_new(k);
				break;
			end
        end
    end
    
    % If no successful choice of k is found.
	if isempty(km)
		km=randi(11);
    end
    
    trainK{i}=km;
    trainS = circshift(trainS,[1 0]);
	label = circshift(label,[1 0]);
    
    clear km distanceI distanceM;
	
end

% For heuristic learning.
minDist=min(nnDist);
maxDist=max(nnDist);

% Testing.
for ii=1:n
    x0=testS(ii, :);
    m2=size(trainS, 1);
    distanceM=zeros(1, m2);
    for f=1:m2
        distanceM(f)=norm(x0-trainS(f, :));
    end
    [distanceM, distanceI]=sort(distanceM, 2, 'ascend');
    nnTestDist=distanceM(1);
    k_med_new(ii)=learningModel(distanceI, trainK, m, nnTestDist, minDist, maxDist, k1_max);
    p(ii)=kNNImb(label, k_med_new(ii), distanceI, classes, pClass);
    
    clear distanceI distanceM;
end

% Index calculation.
accuracy=sum(labelT==p)/length(labelT);
for ii=1:classNum
	classAcc(ii)=sum(labelT(labelT==classes(ii))==p(labelT==classes(ii)))/length(labelT(labelT==classes(ii)));
end
gmeans=nthroot(prod(classAcc), classNum);

end



