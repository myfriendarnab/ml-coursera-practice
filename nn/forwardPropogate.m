function A_Next= forwardPropogate(Theta,A)
    m = size(A, 1);
    A = [ones(m, 1) A];
    A_Next = sigmoid(A*Theta');
end