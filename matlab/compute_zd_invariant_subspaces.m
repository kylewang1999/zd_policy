function [inv_subspace, eigs] = compute_zd_invariant_subspaces(A, z_dim)
% Given a matrix A, compute all z_dim dimensional invariant subspaces
% returns orthogonal bases for each basis

inv_subspace = {};
eigs = {};
[V, D] = eig(A);
lambdas = diag(D);
% Can't handle repeated eigenvalues
if numel(unique(lambdas)) ~= size(D, 1)
    error('Cannot yet compute invariant subspaces for matrices with repeated eigenvalues')
end
real_inds = imag(lambdas) == 0;
comp_inds = imag(lambdas) > 0;
real_inds = find(real_inds);
comp_inds = find(comp_inds);

% First, compute all real subspaces
real_combos = nchoosek(real_inds, z_dim);
for ii = 1:size(real_combos, 1)
    subspace = V(:, real_combos(ii, :));
    inv_subspace = [inv_subspace subspace];
    eigs = [eigs lambdas(real_combos(ii, :))];
end

% Iteratively add a pair of complex eigenvalues until done
for ii = 1:floor(z_dim/2)
    comp_combos = nchoosek(comp_inds, ii);
    for jj = 1:size(comp_combos, 1)
        comp_inv_sub = V(:, comp_combos(jj, :));
        w1 = real(comp_inv_sub);
        w2 = imag(comp_inv_sub);

        real_completion = nchoosek(real_inds, z_dim - 2 * ii);
        for kk = 1:size(real_completion, 1)
            real_sub = V(:, real_completion(ii, :));
            inv_subspace = [inv_subspace [w1 w2 real_sub]];
            eigs = [eigs [lambdas(comp_combos(jj, :)) conj(lambdas(comp_combos(jj, :))) lambdas(real_completion(ii, :))]];
        end
    end
end
end