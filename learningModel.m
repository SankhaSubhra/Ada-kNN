function [K_valT]=learningModel(Idx, k_value, a, d_minY, d_minTr, d_maxTr, K1max)
    
% Heuristic learning algorithm.

    if d_minY<=d_minTr
        k=K1max;
    elseif d_minY>=d_maxTr
        k=1;
    elseif d_minY>d_minTr && d_minY<d_maxTr
        kLin=K1max*exp((d_minY-d_minTr)*(-log(K1max)/(d_maxTr-d_minTr)));
        kExp=(((1-K1max)/(d_maxTr-d_minTr))*(d_minY-d_minTr))+K1max;
        k=ceil(sqrt(kLin*kExp));
    end
    
    Index_of_nearest=Idx(1:k);
    con_arr=[];
    for i=1:k
        con_arr=horzcat(con_arr, k_value{Index_of_nearest(i)});
    end
    max_count=hist(con_arr,1:ceil(sqrt(a)));
    [max_fr,All_possibleK]=max(max_count);
    for p=1:ceil(sqrt(a))
        if max_count(p)==max_fr
            All_possibleK=union(All_possibleK,p);
        end
    end
    pos=randi(numel(All_possibleK));
    K_valT=All_possibleK(pos);
    
end
    

