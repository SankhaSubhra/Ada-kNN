function [p, accuracy, gmeans] = adaKnnGIHS( trainS,label, testS, labelT, k_max)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Implementation of Ada-kNN+GIHS for imbalanced classification problem.
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
k_med=zeros(k_max, m);
p=zeros(n, 1);
classes=unique(label);
classNum=length(classes);
classAcc=zeros(classNum, 1);
emptyCount = 0;
emptyArray = [];
rng('shuffle');
k_Alpha=min(10, k_max);

% GIHS weights calculation.
ideal=ones(1, classNum)./classNum;
actual=hist(label, classes);
actual=actual./m;
zeroActual=(actual==0);
pClass=ones(1, classNum);
change=zeros(1, classNum);
change(~zeroActual)=(ideal(~zeroActual)-actual(~zeroActual))./actual(~zeroActual);
pClass=pClass+change;

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
    [~, distanceI]=sort(distanceM, 2, 'ascend');
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
	if ~isempty(km)
		k_med(km(randi(length(km))), i) = 1;
	else
		emptyCount = emptyCount + 1;
		emptyArray = [emptyArray; i];
	end
   
    trainS = circshift(trainS,[1 0]);
	label = circshift(label,[1 0]);
    
    clear km distanceI distanceM;
	
end
trainS(emptyArray,:) = [];
k_med(:,emptyArray') = [];
label(emptyArray')=[];

% Learning using neural network.
net=patternnet(15); % Set here to update the number of hidden nodes.
net=train(net,trainS',k_med);
k_med_out=net(testS');
[~, k_med_new]=max(k_med_out);

% Testing.
for ii=1:n
    x0=testS(ii, :);
    m2=size(trainS, 1);
    distanceM=zeros(1, m2);
    for f=1:m2
        distanceM(f)=norm(x0-trainS(f, :));
    end
    [~, distanceI]=sort(distanceM, 2, 'ascend');
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



           
   
       
