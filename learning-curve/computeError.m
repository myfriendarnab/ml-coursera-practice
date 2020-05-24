function e = computeError(X,y,theta)
    %compute unregularized error for linear regression
    [e,~] = linearRegCostFunction(X,y,theta,0);
end