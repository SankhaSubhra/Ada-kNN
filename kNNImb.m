function [ pLabel ] = kNNImb( tLabel, k, distanceI, allLabel, pClass )

% Implementation of weighted k-nearest neighbor algorithm.

kNeighbour=distanceI(1:k);
neighbourLabel=tLabel(kNeighbour);
h=hist(neighbourLabel, allLabel);
h=h.*pClass;
[maxH, ~]=max(h);
maxI=(h==maxH);
maxI=maxI.*pClass;
[~, labelPos]=max(maxI);
pLabel=allLabel(labelPos);

end

