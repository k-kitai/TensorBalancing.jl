#
# The code of the function knight_ruiz() in this file is obtained from a open
# source project, HiC-spector (https://github.com/gersteinlab/HiC-spector).
# We modified it slightly so as the function fit to our comparison codes.
#
# Reference
#   Yan KK, Galip Gürkan Yardımcı, William S Noble and Gerstein M.
#   "HiC-Spector: a matrix library for spectral and reproducibility analysis of Hi-C contact maps."
#   Bioinformatics 22 March 2017. https://doi.org/10.1093/bioinformatics/btx152
#
#####################################

#This code was adapted from the MATLAB code implemented in Knight and Ruiz, IMA Journal of Numerical Analysis (2012)
function knight_ruiz(M, tol=1e-6; log_norm=false);
	M[isnan.(M)]=0;
	L=size(M,1);
	iz=find(sum(M,2).>0);
	A=M[iz,iz];
	n=size(A,1);
	e = ones(n,1);
	res=[];
	delta = 0.1;
	# x0 = e;
    x0 = 1 ./ sqrt.(sum(A, 1)')
	#tol = 1e-6;
	g=0.9; etamax = 0.1; # Parameters used in inner stopping criterion.

	eta = etamax;
	x = x0; rt = tol^2; v = x.*(A*x); rk = 1 - v;
	rho_km1=sum(rk.^2);
	rout = rho_km1; rold = rout;
	MVP = 0; # count matrix vector products.
	i = 0; # Outer iteration count.

	while rout > rt # Outer iteration
    	i = i + 1; k = 0; y = e;
    	innertol = maximum([eta^2*rout;rt]);
    	while rho_km1 > innertol #Inner iteration by CG
        	k = k + 1;
        	if k == 1
            	Z = rk./v; p=Z; rho_km1 = sum(rk.*Z);
        	else
            	beta=rho_km1/rho_km2;
            	p=Z + beta*p;
        	end
        	# Update search direction efficiently.
        	w = x.*(A*(x.*p)) + v.*p;
        	#w=squeeze(w,2);
        	alpha = rho_km1/sum(p.*w);
        	ap =Base.squeeze(alpha*p,2);
        	# Test distance to boundary of cone.
        	ynew = y + ap;
        	if minimum(ynew) <= delta
            	if delta == 0
            		break
            	end
            	ind = find(ap .< 0);
            	gamma = minimum((delta - y[ind])./ap[ind]);
            	y = y + gamma*ap;
            	break
        	end
        	y = ynew;
        	rk = rk - alpha*w; rho_km2 = rho_km1; rho_km2=rho_km2[1];
        	Z = rk./v; rho_km1 = sum(rk.*Z);
            log_norm && @printf "rho_km1^0.5=%.13f\n" sqrt(rho_km1)
    	end
    	x = x.*y; v = x.*(A*x);
    	rk = 1 - v; rho_km1 = sum(rk.*rk); rout = rho_km1;
    	MVP = MVP + k + 1;
    	# Update inner iteration stopping criterion.
    	rat = rout/rold; rold = rout; r_norm = sqrt(rout);
    	eta_o = eta; eta = g*rat;
    	if g*eta_o^2 > 0.1
        	eta = maximum([eta;g*eta_o^2]);
    	end
    	eta = maximum([minimum([eta;etamax]);0.5*tol/r_norm]);
    	#@printf("%3d %6d %.3e %.3e %.3e \n", i,k,r_norm,minimum(y),minimum(x));
        #display(rout);
        log_norm && @printf "norm=%.13f\n" sqrt(rout)
        #res=[res; r_norm];
	end
	#@printf("Matrix-vector products = %6d\n", MVP);
	x=Base.squeeze(x,2);
	A2=A*diagm(x);
	A2=diagm(x)*A2;
	A_balance=extend_mat(A2,iz,L);
	A_balance=(A_balance+A_balance')/2;
	x_final=zeros(L);
	x_final[iz]=x;

	return x_final,A_balance;

end

function extend_mat(Z,iz,L);
    (u,v)=ind2sub(size(Z),find(Z.!=0));
    w=Z[find(Z)];
    #w=nonzeros(Z);
    u=iz[u];
    v=iz[v];
    Z_extend=sparse(u,v,w,L,L);
    Z_extend=full(Z_extend);
    return Z_extend;
end
