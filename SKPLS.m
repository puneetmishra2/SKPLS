function [beta, order, count, attributes,model_preps,R,rms] = SKPLS(X, y, A,yadd,plotting,lv_order)
% Swiss Kinfe Partial least sqaure for modelling wide data types
% Reference: https://doi.org/10.1016/j.aca.2022.339786 
% X is the cell containing data blocks such as X = {x1,x2,x3}, In case of
% single block X is just a cell of single block X = {x1}
% y is the response (can be multiresponse or dummy classes for
% classification)
% A is the number of model components
% yadd any addition information realted to samples
% plotting : 1 or 0 to have plots or not
% lv_order : for multiblock scenario user can define the order of lvs
% extraction to achieve sequential models

n  = size(X{1},1); nb = length(X); % Number of samples and blocks
Xtemp = X; % Make a copy of data
for i = 1:nb     % vars in different blocks
    temp_dim = size(Xtemp{1,i});
    Xtemp{1,i} = reshape(Xtemp{1,i},temp_dim(1,1),prod(temp_dim(1,2:end)));
end

pk = cellfun(@(x) size(x,2), Xtemp);   % Number of variables per block
T  = zeros(n,A);     % To be the A orthonormal scores
q  = zeros(size(y,2),A);     % To be the T-regression coeffs
count = zeros(1,nb); % Counts the number of times a block is active
order = zeros(1,A);  % Keeps track of active blocks order
Pb = cell(1,nb);     % Orthonormal block-loadings and -weights:
yorig = y;   % Store original data before centering:
W  = sparse(sum(pk),A); % Global weights
Wb = cellfun(@(x) zeros(x, n), num2cell(pk), 'UniformOutput', false);

for i = 1:nb     % mean centering data
    if ndims(X{1,i})>2
        cc = {[1 0 0; 0 0 0]};
        m_prepX=npreproc(X{1,i},cc{1});
        X{1,i} = m_prepX.Xprep;
        model_preps{1,i} = m_prepX;
    else
        model_preps{1,i} = mean(X{1,i});
        X{1,i} = X{1,i}-mean(X{1,i});

    end
end

Xorig = X;
y  = y - mean(y);     % mean centering response
yprim = y;
yadd = bsxfun(@minus,yadd,mean(yadd));  % mean centering meta info
Y = [yprim yadd];    % concatenating response and meta information

inds  = cellfun(@(x) 1:x, num2cell(pk), 'UniformOutput', false);
for i = 2:nb % Block column-indices
    inds{i} = inds{i} + sum(pk(1:(i-1)));
end

% Temporary (competing) scores and residuals:
t = zeros(n,nb); r = cell(1,nb);
wjx = cell(nb,1);  wkx = cell(nb,1);

accum_ccs = zeros(nb,A);
count_accum = [];

% ------ ROSA solution for single block, multi-block, multi-way, multi-response and CPLS --------%
for a = 1:A
    for k=1:nb
        if ndims(X{1,k})>2         % extracting scores
            dims=size(X{1,k});
            E = reshape(X{1,k}, dims(:,1), prod(dims(:,2:end)));
            [temp_v,~] = Ra(E, Y, yprim, []);
            Zz=reshape(temp_v,dims(:,2:end));
            [wjx{k},~,wkx{k}] = svds(Zz, 1);
            t(:,k) = E*kron(wkx{k},wjx{k});
        else
            [wjx{k},~] = Ra(X{k}, Y, yprim, []);
            wjx{k}(abs(wjx{k})<eps) = 0;                % Removes insignificant values
            t(:,k) = X{k}*wjx{k};
        end
    end

    if a > 1   % normalise the competing scores with
        for k = 1:nb
            t(:,k) = t(:,k)- T(:,1:a-1)*(T(:,1:a-1))'*t(:,k);
        end
    end

    for k=1:nb
        t(:,k) = t(:,k)/norm(t(:,k));    % Normalize the competing scores
        r{k} = yprim - t(:,k)*(t(:,k)'*yprim); % .. and calculate y-residuals
    end

    for k=1:nb  % estimating canonical correaltions to select winners
        [~,c] = Ra(t(:,k),yprim, yprim, []);
        accum_ccs(k,a) = c;
    end

    temp_count = zeros(1,nb);
    if ~exist("lv_order")
        [~,i] = max(accum_ccs(:,a)); % for multiresppnse use the canonical correlation
    else
        i = lv_order(1,a); % for SO-PLS solution
    end

    temp_count(:,i) = 1;
    count_accum = [count_accum ;temp_count];
    count(i) = count(i)+1;  % Book-keeping (record counts and
    order(a) = i;           % .. order of winning blocks)
    T(:,a)   = t(:,i);      % The winning score-vector is stored
    q(:,a)   = y'*T(:,a);   % Regression coeff wrt T(:,a)
    yprim = r{i};      % yprim is updated to the smallest residual
    Y = [yprim yadd];  % y for CPLS is also updated for single block analysis


    % Orthogonalize and normalize the winning weights
    if ndims(X{1,i})>2
        www = kron(wkx{i},wjx{i});
        www = www - Wb{i}(:,1:count(i))*(Wb{i}(:,1:count(i))'*www);
        Wb{i}(:,count(i)) = www/norm(www);
        clear www
        W(inds{1,i},a) = Wb{i}(:,count(i)) ;
    else
        wjx{i} = wjx{i} - Wb{i}(:,1:count(i))*(Wb{i}(:,1:count(i))'*wjx{i});
        Wb{i}(:,count(i)) = wjx{i}/norm(wjx{i});
        W(inds{1,i},a) = Wb{i}(:,count(i));
    end
end

%%%%%% post processing to extract all relevant model parameters %%%%%%%%%%
for k = 1:nb    % Pb estimation
    if ndims(X{k})>2
        dims = size(X{k});
        temp_data = reshape(X{k},dims(:,1),prod(dims(:,2:end)));
        Pb{k} = temp_data'*T;
    else
        Pb{k} = X{k}'*T;
    end
end

PtW   = triu(cell2mat(Pb')'*W);    % PtW estimation

for k = 1:nb   % reshaping multiway data for offset estimaiton for beta later
    if ndims(Xorig{k}>2)
        dims = size(Xorig{k});
        Xorig{k} = reshape(Xorig{k},dims(1,1),prod(dims(:,2:end)));
    end
end

for i = 1:size(y,2)  % estimation of beta for future use
    temp_beta  = cumsum(bsxfun(@times,W/PtW, q(i,:)),2);
    beta{1,i}  = [mean(yorig(:,i)) - mean(cell2mat(Xorig))*temp_beta; temp_beta];
    beta{1,i}  = beta{1,i}(:,end);
end

for i = 1:size(y,2)
    [R(i,:),rms(i,:)]=rmse(yorig(:,i),([zeros(n,1) cell2mat(Xorig)]*beta{i})+mean(yorig(:,i)));
end
if plotting ==1   % some plots for understading the results
    for i = 1:size(y,2)
        subplot(1,size(y,2),i)
        plot(yorig(:,i),([zeros(n,1) cell2mat(Xorig)]*beta{i})+mean(yorig(:,i)),'or');xlabel('Measured');ylabel('Predicted');
        lsline;
        title(['SKPLS LVs = ' num2str(A) ' R = ' num2str(round(R(i,:),2)) ' RMSE = ' num2str(round(rms(i,:),2))]);
    end
    figure,
    subplot(1,2,1)
    bar(count,'stacked');ylabel('LVs');
    for i = 1:nb
        names{i} = ['Block ' num2str(i)];
    end
    set(gca,'xtick',[1:i],'xticklabel',names);
    xtickangle(30)
    subplot(1,2,2)
    imagesc(cumsum(count_accum,1));ylabel('Order of selection (Read top to bottom)');
    set(gca,'xtick',[1:i],'xticklabel',names);
    xtickangle(30)
end

% Remove unused Wb-columns:
for k=1:nb
    Wb{k} = Wb{k}(:,1:count(k));
end
% Collect attributes
attributes = struct('T', T, 'Pb', {Pb}, 'Wb', {Wb}, 'W', W, 'PtW', PtW);
end


%%%%%%%%%%%%%%%%%%some auxilary functions%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function model=applnpreproc(X,nprmodel)
% storing the dimensions of the array
dimX=size(X);
preppars=nprmodel.preppars;
preproc=nprmodel.prepropt;
Xp=X;

%loop across the different modes ('scale first')
for i=1:3
    pord=[i:3 1:i-1];
    Xp=permute(Xp, pord);
    dimp=dimX(pord);
    if preproc(2,i)==1
        xu=reshape(Xp,dimp(1), dimp(2)*dimp(3));
        s=preppars{2,i};
        xp=xu./repmat(s,1,dimp(2)*dimp(3));
        Xp=reshape(xp,dimp(1),dimp(2),dimp(3));
    end
    Xp=ipermute(Xp, pord);
end

for i=1:3
    pord=[i:3 1:i-1];
    Xp=permute(Xp, pord);
    dimp=dimX(pord);
    if preproc(1,i)==1
        xu=reshape(Xp,dimp(1),dimp(2)*dimp(3));
        m=preppars{1,i};
        xp=xu-repmat(m,dimp(1),1);
        Xp=reshape(xp,dimp(1),dimp(2),dimp(3));
    end
    Xp=ipermute(Xp, pord);
end
model.Xprep=Xp;
model.Xraw=X;
model.prepropt=nprmodel.prepropt;
model.preppars=nprmodel.preppars;
model.scalefact=nprmodel.scalefact;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [R,rms]=rmse(y,yhat)
%rmse Root-mean-square error.
R=corr(y(:),yhat(:));
rms=sqrt(sum((y(:)-yhat(:)).^2)/length(y));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function model=npreproc(X,preproc,varargin)
if isempty(varargin)
    scalefact='mse';
else
    scalefact=varargin{1};
end
% storing the dimensions of the array
dimX=size(X);
preppars=cell(size(preproc));
Xp=X;
%loop across the different modes ('scale first')
for i=1:3
    pord=[i:3 1:i-1];
    Xp=permute(Xp, pord);
    dimp=dimX(pord);
    if preproc(2,i)==1
        xu=reshape(Xp,dimp(1), dimp(2)*dimp(3));
        switch scalefact
            case 'std'
                s=std(xu, [],2);
            case 'mse'
                s=sqrt(sum(xu.^2,2)./(dimp(2)*dimp(3)));
        end

        xp=xu./repmat(s,1,dimp(2)*dimp(3));
        Xp=reshape(xp,dimp(1),dimp(2),dimp(3));
        preppars{2,i}=s;
    end
    Xp=ipermute(Xp, pord);
end

for i=1:3
    pord=[i:3 1:i-1];
    Xp=permute(Xp, pord);
    dimp=dimX(pord);
    if preproc(1,i)==1
        xu=reshape(Xp,dimp(1),dimp(2)*dimp(3));
        m=mean(xu);
        xp=xu-repmat(m,dimp(1),1);
        Xp=reshape(xp,dimp(1),dimp(2),dimp(3));
        preppars{1,i}=m;
    end
    Xp=ipermute(Xp, pord);
end
model.Xprep=Xp;
model.Xraw=X;
model.prepropt=preproc;
model.preppars=preppars;
model.scalefact=scalefact;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Ra function (for CPLS excluding powers)
function [w,cc] = Ra(X, Y, Yprim, wt)
W = X'*Y;
[r,A] = ccXY(X*W, Yprim, wt); % Computaion of canonical correlations between
% XW and Y with the rows weighted according to wt.
w     = W*A(:,1);             % The optimal loading weight vector
cc    = r(1)^2;               % squared canonical correlation
end

%%%%%%%%%%%%%%%%%%%% Canonical correlations
function [r,A] = ccXY(X,Y,wt)
% Computes the coefficients in canonical variates between collumns of X and Y
[n,p1] = size(X); p2 = size(Y,2);

% Weighting of observations with regards to wt (asumes weighted centering already performed)
if ~isempty(wt)
    X = rscale(X,wt);
    Y = rscale(Y,wt);
end

% Factoring of data by QR decomposition and ellimination of internal linear
% dependencies in X and Y
[Q1,T11,perm1] = qr(X,0);       [Q2,T22,~] = qr(Y,0);
rankX          = sum(abs(diag(T11)) > eps(abs(T11(1)))*max(n,p1));
rankY          = sum(abs(diag(T22)) > eps(abs(T22(1)))*max(n,p2));
if rankX < p1
    Q1 = Q1(:,1:rankX); T11 = T11(1:rankX,1:rankX);
end
if rankY < p2
    Q2 = Q2(:,1:rankY);
end

% Economical computation of canonical coefficients and canonical correlations
d = min(rankX,rankY);
if nargout == 1
    D    = svd(Q1' * Q2,0);
    r    = min(max(D(1:d), 0), 1); % Canonical correlations
else
    [L,D]    = svd(Q1' * Q2,0);
    A        = T11 \ L(:,1:d) * sqrt(n-1);
    % Transform back coefficients to full size and correct order
    A(perm1,:) = [A; zeros(p1-rankX,d)];
    r = min(max(diag(D(1:d)), 0), 1); % Canonical correlations
end

end

%%%%%%%%%%%%%%%%% Weighted centering
function [X, mX, n, p] = Center(X,wt)
% Centering of the data matrix X by subtracting the weighted column means
% according to the nonegative weights wt
[n,p] = size(X);
% Calculation of column means:
if nargin == 2 && ~isempty(wt)
    mX = (wt'*X)./sum(wt);
else
    mX = mean(X);
end
% Centering of X, similar to: %X = X-ones(n,1)*mX;
X = X-repmat(mX,n,1);
end

function X = rscale(X,d)
% Scaling the rows of the matrix X by the values of the vector d
X = repmat(d,1,size(X,2)).*X;
end

