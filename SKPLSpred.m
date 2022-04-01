function [R,rmsep] = SKPLSpred(Xt,yt,Xc,yc,prop,beta,preps, plotting)
% for making prediction for nROSA models
nb = length(preps);
n = size(Xt{1,1},1);
for i = 1:nb     % mean centering data
        if ndims(Xt{1,i})>2
           model=applnpreproc(Xt{1,i},preps{1,i});
           Xt{1,i} = model.Xprep;
           model=applnpreproc(Xc{1,i},preps{1,i});
           Xc{1,i} = model.Xprep;
        else
           Xt{1,i} = Xt{1,i}-preps{1,i};     
           Xc{1,i} = Xc{1,i}-preps{1,i};   
        end
end

for k = 1:nb
    if ndims(Xt{1,k}>2)
       dims = size(Xt{1,k});
       Xt{1,k} = reshape(Xt{1,k},dims(1,1),prod(dims(:,2:end)));
       dims = size(Xc{1,k});
       Xc{1,k} = reshape(Xc{1,k},dims(1,1),prod(dims(:,2:end)));
    end
end

[R,rmsep]=rmse(yt,([zeros(size(yt,1),1) cell2mat(Xt)]*beta(:,end))+mean(yc));
[Rc,rmsec]=rmse(yc,([zeros(size(yc,1),1) cell2mat(Xc)]*beta(:,end))+mean(yc));

if plotting==1
    figure,
    plot(yc,([zeros(size(yc,1),1) cell2mat(Xc)]*beta(:,end))+mean(yc),'ob');hold on;
    plot(yt,([zeros(size(yt,1),1) cell2mat(Xt)]*beta(:,end))+mean(yt),'^r');xlabel(['Measured ' prop]);ylabel(['Predicted' prop]);
    lsline;
    title(['R_c = ' num2str(round(Rc,2)) ' RMSEC = ' num2str(round(rmsec,2)) ' R_p = ' num2str(round(R,2)) ' RMSEP = ' num2str(round(rmsep,2))]);
    legend('Calibration','Test set');
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [R,rms]=rmse(y,yhat)
%rmse Root-mean-square error.
%   [R,rms] = rmse(y,yhat) computes the correlation coefficient (R) and
%   root mean square error (rms) betwwen the reference value (y) and predicted
%   value (yhat). R and rms are both scales.
%   Copyright Zhang Jin (zhangjin@mail.nankai.edu.cn).
R=corr(y(:),yhat(:));
rms=sqrt(sum((y(:)-yhat(:)).^2)/length(y));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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