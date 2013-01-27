% Multi level Monte Carlo integrator
% by Juho Häppölä
% juho.happola@iki.fi
%
% Code can be used in terms of
% Creative commons
% Attribution-ShareAlike 3.0 Unported

%%% insert definitions that will become handy %%%
Phi = @(x) 0.5*(1+erf(x/sqrt(2)));  % CDF for standard gaussian
d1 = @(S0,K,sig,T,r) (log(S0/K)+(r+0.5*sig^2)*T)/(sig*sqrt(T));
d2 = @(S0,K,sig,T,r) d1(S0,K,sig,T,r)-sig*sqrt(T);
fbs = @(S0,K,sig,T,r) Phi(d1(S0,K,sig,T,r))*S0 - Phi(d2(S0,K,sig,T,r))*K*exp(-1*r*T);

TOL = 1e-2;

molpar = 1e-12*TOL;
H = @(x) Phi(x/molpar);

emptysparse = @(x,y) speye(x,y)-speye(x,y); %kludge...

%%% end of auxiliary misc. stuff %%%

tc = clock;

L = 4 % the maximum level
m1 = 1e4 % number of paths on the coarsest grid
m2 = 2 % number of time steps on the coarsest grid
m3 = 2 % how much to refine
dim = 5 % dimension
T = 1/2 % final time

X_0 = 100*ones(dim,1); % initial value

r=1e-2         %interest rate
sig = magic(dim); %volatility
sig = sig+sig';
sig = 1/norm(sig)*sig*0.1;
K = mean(X_0) %strike

[su sv] = eig(sig);

a = @(x,t) x.*r %drift part
%a = @(x,t) x.*r.*(1+0.5*cos(50*t));
%b = @(x,t) sig.*x % volatility part
%b = @(x,t) sig.*x.*(1+0.5*cos(50*t))


b = @(x,t) sig*x
g = @(x,t) exp(-r*T)*max(mean(x(:,end))-K,0);



xmin = 1e-1;
xmax = 1.5;
xvals = linspace(xmin,xmax,15);    % values at which to evaluate the CDF that is being estimated

nlev = zeros(L,1);  % this vector accumulates the info on how many realisations are generated on level l
nlev(1)= m1;  % all the paths are at least first level

gmat = emptysparse(m1*L,length(xvals));

%gmat is a sparse matrix that contains "all the essential information"
%for technical reasons this is implemented as a 2d array
%gmat(1:nlev(l),l) contains the realisations for g_0
%below these, there are realisations for g_l-g_{l-1}
%stored at the rows gmat(1+(l-1)*m1:nlev(l)+(l-1)*m1,l)

maxpaths = 5e3;
% this is the limit on how many paths are integrated at once
% too small a value will cause slowdown in the form of looping
% too large one will cause the same because of swapping

%matlabpool % seems to run faster with the pool set on

%%% this is only for the first level %%%%

numIter = m1;
generatedPaths = 0;
%gmatLevel1 = emptysparse(m1,length(xvals));
parfor aiter = [1:numIter]

    ts_l = linspace(0,T,m2+1); % time steps
    increments = [zeros(dim,1) randn(dim,m2)]; %the actual randomness is here!!!
    dt = ts_l(2)-ts_l(1); % time step
    Ws = cumsum(sqrt(dt)*increments,2);  % generate the Wiener paths from the increments
    hs = size(Ws); % helper
    Xs = Ws; % the actual trajectories
    Xs(:,1) = X_0; % set the initial value
    for ts =[2:hs(2)]
		Xs(:,ts) = Xs(:,ts-1)+a(Xs(:,ts-1),ts_l(ts-1))*dt; % drift
		Xs(:,ts) = Xs(:,ts)+b(Xs(:,ts-1),ts_l(ts-1)).*(Ws(:,ts)-Ws(:,ts-1)); %volatility
    end
	tempg = [1:length(xvals)];
    for obs=[1:length(xvals)]
        G = @(x,t) H(xvals(obs)-g(x,t)); %generate a QOI G from g and the actual payoff g
		tempg(obs) = G(Xs,ts_l);
    end
	gmat(aiter,:) = tempg;
end

sprintf('Coarsest grid (g_0) done')
%%% %%%

%%% the higher order terms

for l=[2:L]
M_l = max(ceil(m1*(2*m3)^(1-l)),2);
numIter = M_l;
nlev(l) = M_l;
parfor aiter=[1:numIter]

    ts_l = linspace(0,T,m2*(2*m3)^(l-1)+1); % the time steps on this level
    increments = [zeros(dim,1) randn(dim,m2*(2*m3)^(l-1))]; % once again, here's the randomness
    dt = ts_l(2)-ts_l(1); % time step
    Ws = cumsum(sqrt(dt)*increments,2); % construct the wiener process
    hs = size(Ws); %helper 
    Xs = Ws; %Xs to be integrated
    Xs(:,1) = X_0;
    for ts =[2:hs(2)]
     % the actual integration
     Xs(:,ts) = Xs(:,ts-1)+a(Xs(:,ts-1),ts_l(ts-1))*dt;
     Xs(:,ts) = Xs(:,ts)+b(Xs(:,ts-1),ts_l(ts-1)).*(Ws(:,ts)-Ws(:,ts-1));
    end
    
    jump = 2*m3; % jump over this many time steps
    ts_eff = ts_l(1:jump:end);
    dt = jump*dt; % interpolated time step
    Ws_eff = Ws(:,1:jump:end); % take only the relevant part of the randomness
    Xs_eff = Ws_eff;
    Xs_eff(:,1) = X_0;
    hs = size(Ws_eff);
    for ts=[2:hs(2)]
        % FE integration
        Xs_eff(:,ts) = Xs_eff(:,ts-1)+a(Xs_eff(:,ts-1),ts_eff(ts-1))*dt;
        Xs_eff(:,ts) = Xs_eff(:,ts)+b(Xs_eff(:,ts-1),ts_eff(ts-1)).*(Ws_eff(:,ts)-Ws_eff(:,ts-1));
    end
	tempg = [1:length(xvals)]
    for obs=[1:length(xvals)]
        G = @(x,t) H(xvals(obs)-g(x,t));
		tempg(obs) = G(Xs,ts_l) - G(Xs_eff,ts_eff);
    end
	gmat(aiter,:) = tempg
end
sprintf('Level %d done',l)
end

CDFvals = [];
variances = zeros(L,length(xvals));

for obs=[1:length(xvals)]
    zerolevel = mean(gmat(1:m1,obs));
    corrections = 0;
    for level=[2:L]
        levelShift = m1*(level-1);
       corrections = corrections + mean(gmat(1+levelShift:nlev(level)+levelShift,obs));
    end
    CDFvals = [CDFvals zerolevel+corrections];
end

%%%% the error estimation and analysis of results starts here %%%%

nsig = 2;
estimates = zeros(length(xvals),L);

for obs=[1:length(xvals)]
   estimates(obs,1) = mean(gmat(1:m1,obs));
   for level = [2:L]
      levelShift = m1*(level-1);
      estimates(obs,level) = estimates(obs,level-1) + mean(gmat(1+levelShift:nlev(level)+levelShift,obs));
   end
end

variances = zeros(length(xvals),L);

for obs=[1:length(xvals)]
   variances(obs,1) = var(gmat(:,obs))/m1;
   for level = [2:L]
      levelShift = m1*(level-1);
      variances(obs,level) = variances(obs,level-1) + var(gmat(1+levelShift:nlev(level)+levelShift,obs))/nlev(level);
   end
end

staterrs = sqrt(cumsum(nsig*variances,2));

%this contains the statistical error up to a certain level
%each row corresponds to a given observable
%and each column to a choice of L.

rerrs = zeros(length(xvals),L);
for obs=[1:length(xvals)]
    for level = [2:L]
	%levelShift = m1*(level-1);
	precoef = 1/abs(1-1/(2*m3));
	rerrs(obs,level) = precoef*abs(mean(gmat(1+levelShift:nlev(level)+levelShift,obs)));
	%rerrs(obs,level) = abs(mean(gmat(1+levelShift:nlev(level)+levelShift)))/(1-1/sqrt(2*m3));
    end
end



totalerrs = staterrs+rerrs;
%%%% 

CDF_values = max(min(estimates(:,end),1),0);

psi_doubleprime = 0;

for h1=[1:length(CDF_values)-2]
for h2=[h1+1:length(CDF_values)-1]
for h3=[h2+1:length(CDF_values)]
psi_doubleprime = max(psi_doubleprime, abs((CDF_values(h3)-CDF_values(h2))/(xvals(h3)-xvals(h2))-(CDF_values(h2)-CDF_values(h1))/(xvals(h2)-xvals(h1))));
end
end
end

psi_doubleprime = full(max(abs(diff(diff(CDF_values))))*(mean(diff(xvals)))^(-2))


plotn = 2e2;

plotx = linspace(xmin,xmax,plotn);
refined_CDF_values = interp1(xvals,CDF_values,plotx);
refined_pointwise_errors = interp1(xvals,totalerrs(:,end),plotx);
interperror = zeros(1,length(plotx));

for plotind = [1:length(plotx)]
interperror(plotind) = min(abs(plotx(plotind)-xvals))^2;
end

interperror = interperror*psi_doubleprime*1/8;

figure;
hold on;
lw = 1
errorbar(xvals,CDF_values,totalerrs(:,end));
plot(plotx,refined_CDF_values,'b','LineWidth',lw);
plot(plotx,min(refined_CDF_values+refined_pointwise_errors+interperror,1),'r--','LineWidth',lw);
plot(plotx,max(refined_CDF_values-refined_pointwise_errors-interperror,0),'r--','LineWidth',lw);

grid on

xlabel 'x'
ylabel '\Psi (x)'

stat_error = max(staterrs(:,end))
bias_error = max(rerrs(:,end))
interperror = max(interperror)
total_error = max(abs(refined_pointwise_errors+interperror))
TOL


maxrerrs = max(rerrs)*(abs(1-1/(2*m3)));
hC = abs((T/m2)*(1-2*m3))*maxrerrs(2);
suggested_L = ceil(L - log(TOL/3/bias_error)/log(2*m3))
varg0 = max(staterrs(:,1));
varg1 = max(staterrs(:,2)-staterrs(:,1));
suggested_m1 = m1*stat_error^2*9/TOL/TOL
deltax = sqrt(8*TOL/3*psi_doubleprime);
%suggested_N = ceil((max(xvals)-min(xvals))/deltax)
interperror = psi_doubleprime*max(diff(xvals))^2/8;
suggested_N = ceil(sqrt(3*interperror*length(xvals)^2/TOL))


clear finestep increments W t W_eff t_eff dt l l2 k path gp hs Xs gs Ws
clear zerolevel corrections tv plotn plotx Ws_eff Xs_eff obs ts_l ts_eff
clear variances ts level jump lw

tc2 = clock;
sprintf('Done. Time elapsed %d seconds.',ceil(etime(tc2,tc)))

clear tc tc2


 
