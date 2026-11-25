"""Three-parameter score fusion, fitted only on source dev, and matched calibration control."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scripts.hierarchical_score_fusion_GRU_Temproal_Attention.common import require,status

EPSILON=1e-6

def design(artifact,emotion=None):
    a=np.asarray(artifact,np.float64)
    require(a.ndim==1 and len(a)>0,'Expected nonempty 1D scores')
    values=[a] if emotion is None else [a,np.asarray(emotion,np.float64)]
    cols=[]
    for p in values:
        require(p.shape==a.shape and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid score')
        p=np.clip(p,EPSILON,1-EPSILON)
        cols.append(np.log(p)-np.log1p(-p))
    return np.column_stack(cols+[np.ones(len(a))])

def fit(artifact,emotion,labels,*,split,regularization=.01):
    require(split=='dev','Fusion fitting must use source dev, never test')
    require(regularization>0,'Regularization must be positive')
    x=design(artifact,emotion);y=np.asarray(labels,np.float64)
    require(y.shape==(len(x),) and set(y.tolist())=={0.,1.},'Need both binary dev classes')
    prior=np.zeros(x.shape[1]);prior[0]=1.
    def objective(w):
        z=x@w;delta=w-prior
        return (float(np.mean(np.logaddexp(0,z)-y*z)+.5*regularization*np.dot(delta,delta)),
                x.T@(expit(z)-y)/len(y)+regularization*delta)
    history=[]
    def callback(w):
        value,_=objective(w)
        history.append(dict(iteration=len(history)+1,objective=value,weights=w.tolist()))
        status('FUSION',f'iteration={len(history)} objective={value:.8f}')
    opt=minimize(objective,prior,jac=True,method='L-BFGS-B',callback=callback,
                 options=dict(maxiter=1000,ftol=1e-12,gtol=1e-9))
    require(opt.success and np.isfinite(opt.x).all(),'Score optimization failed: '+str(opt.message))
    return dict(format='trainable_final_score_v1',weights=opt.x.tolist(),emotion_used=emotion is not None,
                epsilon=EPSILON,regularization=regularization,prior=prior.tolist(),
                n=len(y),split=split,label='clip_label',class_weight=None,threshold=.5,
                objective=float(opt.fun),iterations=int(opt.nit),success=bool(opt.success),history=history)

def predict(model,artifact,emotion=None):
    require(model['format']=='trainable_final_score_v1','Unexpected fusion format')
    require(model['emotion_used']==(emotion is not None),'Fusion input mismatch')
    require(model['epsilon']==EPSILON,'Fusion clamp mismatch')
    w=np.asarray(model['weights'])
    x=design(artifact,emotion)
    require(w.shape==(x.shape[1],) and np.isfinite(w).all(),'Invalid fusion weights')
    return expit(x@w)
