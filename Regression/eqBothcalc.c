/*==========================================================
 *
 * construct Phi and predPhi^T using this function
 *
 * Compile as below
 *
 * FOR UNIX
 * mex CC=gcc LD=gcc COPTIMFLAGS='-O3 -DNDEBUG -ffast-math' CFLAGS='\$CFLAGS -fopenmp' LDFLAGS='\$LDFLAGS -fopenmp' eqBothcalc.c
 *
 * FOR OSX
 * follow instructions in openMP_OSX
 * Then build mex file using mex eqBotchcalc.c
 *
 *========================================================*/

#include "mex.h"
#include "math.h"
#include "omp.h"

/* The computational routine */
void eqBothcalc(double *nn_obs, double *nn_pred, double *nn_m, double *basnrs,
        double *obs, double *X, double *Y, double *Lxx, double *Lyy, double *AA,
        double *BB, double *nrSegments, double *addPrevSegments, double *Phi, double *predPhi_T)
{
    
    int n_obs=(int)nn_obs[0],n_pred=(int)nn_pred[0],n_m=(int)nn_m[0],ss,tt,qq,zz,outerInd;
    double Lx=Lxx[0], Ly=Lyy[0], A=AA[0], B=BB[0];
    
    if (n_obs>n_pred) {
        outerInd=n_obs;
    }else{
        outerInd=n_pred;
    }
    
    omp_set_num_threads(omp_get_max_threads()); /* automatically use the maximum number of available threads */
    #pragma omp parallel for private(ss,tt,qq,zz)
    for (ss=0; ss<outerInd; ss++) {
        if (ss<n_obs) {
            int compSegs=(int)addPrevSegments[ss];
            double L1 = sqrt(  (obs[1+4*(ss+compSegs)]-obs[4*(ss+compSegs)])*(obs[1+4*(ss+compSegs)]-obs[4*(ss+compSegs)]) +
                    (obs[3+4*(ss+compSegs)]-obs[2+4*(ss+compSegs)])*(obs[3+4*(ss+compSegs)]-obs[2+4*(ss+compSegs)]) );
            double x01 = obs[4*(ss+compSegs)], y01=obs[2+4*(ss+compSegs)],
                    nx=(obs[1+4*(ss+compSegs)]-obs[4*(ss+compSegs)])/L1,   /* (x1-x0)/L */
                    ny=(obs[3+4*(ss+compSegs)]-obs[2+4*(ss+compSegs)])/L1; /* (y1-y0)/L */
            double totL=0;
            for (zz=0; zz<nrSegments[ss]; zz++) {
                /* calculate L=sqrt( (x1-x0)^2 + (y1-y0)^2 ); */
                double L = sqrt(  (obs[1+4*(ss+compSegs+zz)]-obs[4*(ss+compSegs+zz)])*(obs[1+4*(ss+compSegs+zz)]-obs[4*(ss+compSegs+zz)]) +
                        (obs[3+4*(ss+compSegs+zz)]-obs[2+4*(ss+compSegs+zz)])*(obs[3+4*(ss+compSegs+zz)]-obs[2+4*(ss+compSegs+zz)]) );
                double x0 = obs[4*(ss+compSegs+zz)], y0=obs[2+4*(ss+compSegs+zz)];
                double Lstart=sqrt( (x0-x01)*(x0-x01) + (y0-y01)*(y0-y01) ), Lend=Lstart+L;
                totL = totL+L;
                
                for (tt=0; tt<n_m; tt++) {
                    if (zz==0)
                        Phi[ss+n_obs*tt]=0;
                    double lambdaXin=0.5*M_PI*basnrs[tt]/Lx, lambdaYin=0.5*M_PI*basnrs[tt+n_m]/Ly;
                    double lambdaX=nx*lambdaXin, lambdaY=ny*lambdaYin,
                            BX=(x01+Lx)*lambdaXin, BY=(y01+Ly)*lambdaYin;
                    double lambda_min=lambdaX-lambdaY, B_min=BX-BY,
                            lambda_plus=lambdaX+lambdaY, B_plus=BX+BY;
                    double superConst=nx*nx*(lambdaXin*lambdaXin-A*lambdaYin*lambdaYin)+
                            ny*ny*(lambdaYin*lambdaYin-A*lambdaXin*lambdaXin),
                            otherConst=2*B*nx*ny*lambdaXin*lambdaYin;
                    double theInt = superConst*( (sin(lambda_min*Lend+B_min)-sin(lambda_min*Lstart+B_min))/lambda_min +
                            (-sin(lambda_plus*Lend+B_plus)+sin(lambda_plus*Lstart+B_plus))/lambda_plus )+
                            otherConst*( (sin(lambda_min*Lend+B_min)-sin(lambda_min*Lstart+B_min))/lambda_min +
                            (sin(lambda_plus*Lend+B_plus)-sin(lambda_plus*Lstart+B_plus))/lambda_plus );
                    Phi[ss+n_obs*tt]+=theInt;
                    if (zz==nrSegments[ss]-1) 
                        Phi[ss+n_obs*tt]/=2*totL*(sqrt(Ly*Lx));
                }
            }  
        }
        
        if (ss<n_pred) {
            for (qq=0; qq<n_m; qq++) {
                double lambdaXin=0.5*M_PI*basnrs[qq]/Lx, lambdaYin=0.5*M_PI*basnrs[qq+n_m]/Ly;
                double dx2=-lambdaXin*lambdaXin*(1/sqrt(Ly*Lx))*sin(lambdaXin*(X[ss]+Lx))*
                        sin(lambdaYin*(Y[ss]+Ly));
                double dy2=-lambdaYin*lambdaYin*(1/sqrt(Ly*Lx))*sin(lambdaXin*(X[ss]+Lx))*
                        sin(lambdaYin*(Y[ss]+Ly));
                double dxdy=lambdaYin*lambdaXin*(1/sqrt(Ly*Lx))*cos(lambdaXin*(X[ss]+Lx))*
                        cos(lambdaYin*(Y[ss]+Ly));
                predPhi_T[3*ss+3*n_pred*qq] = A*dy2-dx2;
                predPhi_T[3*ss+1+3*n_pred*qq] = B*dxdy;
                predPhi_T[3*ss+2+3*n_pred*qq] = A*dx2-dy2;
            }
        }
        
    }
    
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    /* inputs */
    double *n_obs;  /* # observations */
    double *n_pred; /* # predictions */
    double *n_m;    /* # basis functions */
    double *basnrs;   /* eigenvalues */
    double *obs;    /* observed integrals */
    double *X;      /* X-pred */
    double *Y;      /* Y-pred */
    double *Lx;     /* x-expansion */
    double *Ly;     /* y-expansion */
    double *A;      /* A-constant (=a/b) */
    double *B;      /* B-constant (=-(a+b)/b) */
    double *nrSegments; /* # segments for each measurement */
    double *addPrevSegments;
    
    /* outputs */
    double *Phi;
    double *predPhi_T;
    
    /* get the value of the scalar input  */
    n_obs = mxGetPr(prhs[0]);
    n_pred = mxGetPr(prhs[1]);
    n_m = mxGetPr(prhs[2]);
    basnrs = mxGetPr(prhs[3]);
    obs = mxGetPr(prhs[4]);
    X = mxGetPr(prhs[5]);
    Y = mxGetPr(prhs[6]);
    Lx = mxGetPr(prhs[7]);
    Ly = mxGetPr(prhs[8]);
    A = mxGetPr(prhs[9]);
    B = mxGetPr(prhs[10]);
    nrSegments = mxGetPr(prhs[11]);
    addPrevSegments = mxGetPr(prhs[12]);
    
    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix( (int)n_obs[0], (int)n_m[0], mxREAL);
    plhs[1] = mxCreateDoubleMatrix( (int)3*n_pred[0], (int)n_m[0], mxREAL);
    
    /* get a pointer to the real data in the output matrix */
    Phi = mxGetPr(plhs[0]);
    predPhi_T = mxGetPr(plhs[1]);
    
    /* call the computational routine */
    eqBothcalc(n_obs,n_pred,n_m,basnrs,obs,X,Y,Lx,Ly,A,B,nrSegments,addPrevSegments,Phi,predPhi_T);
}