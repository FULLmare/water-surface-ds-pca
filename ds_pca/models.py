"""Final water-surface DS+PCA model: three components, ridge 0.001."""
import numpy as np

from ._core import KEY, db, kernel, kernel_features

LOWER=np.array([-80,np.log(.001),np.log(.001),np.log(.001),np.log(.25),np.log(.25),-np.pi/4])
UPPER=np.array([80,np.log(2000),np.log(2000),np.log(2000),np.log(4),np.log(4),np.pi/4])


def log_coordinates(theta,phi,ti):
    t,p,inc=np.deg2rad(theta[:,None]),np.deg2rad(phi[None,:]),np.deg2rad(ti)
    dot=np.cos(t)*np.cos(inc)+np.sin(t)*np.sin(inc)*np.cos(p)
    psi=np.arccos(np.clip(dot,-1,1))
    scale=np.divide(psi,np.sin(psi),out=np.ones_like(psi),where=psi>1e-10)
    u=scale*(np.sin(t)*np.cos(p)*np.cos(inc)-np.cos(t)*np.sin(inc))
    v=scale*np.sin(t)*np.sin(p)
    return u.ravel(),v.ravel()


def extended_value_jac(params,u,v,jac=True):
    g0=params[0];ap,am,av=np.exp(params[1:4]);pt,pv=np.exp(params[4:6]);rot=params[6]
    ur=np.cos(rot)*u+np.sin(rot)*v
    vr=-np.sin(rot)*u+np.cos(rot)*v
    plus=np.maximum(ur,0);minus=np.maximum(-ur,0);absv=abs(vr)
    tp=ap*plus**pt;tm=am*minus**pt;tv=av*absv**pv
    val=g0-tp-tm-tv
    if not jac:return val
    j=np.empty((len(u),7));j[:,0]=1;j[:,1]=-tp;j[:,2]=-tm;j[:,3]=-tv
    j[:,4]=-pt*(tp*np.log(np.maximum(plus,1e-30))+tm*np.log(np.maximum(minus,1e-30)))
    j[:,5]=-pv*tv*np.log(np.maximum(absv,1e-30))
    du=np.where(ur>1e-12,ap*pt*np.maximum(plus,1e-12)**(pt-1),0)-np.where(ur< -1e-12,am*pt*np.maximum(minus,1e-12)**(pt-1),0)
    dv=np.where(absv>1e-12,av*pv*np.maximum(absv,1e-12)**(pv-1)*np.sign(vr),0)
    j[:,6]=-du*vr+dv*ur
    return val,j


def lm_fit(initial,u,v,y,weight,iterations=65):
    p=np.clip(initial,LOWER,UPPER).copy();damping=.03
    accepted=0
    for it in range(iterations):
        value,jac=extended_value_jac(p,u,v)
        error=value-y;loss=np.sum(error**2*weight)
        h=jac.T@(jac*weight[:,None]);g=jac.T@(error*weight)
        step=np.linalg.solve(h+damping*np.diag(np.maximum(np.diag(h),1e-6)),-g)
        candidate=np.clip(p+step,LOWER,UPPER)
        newloss=np.sum((extended_value_jac(candidate,u,v,False)-y)**2*weight)
        if newloss<loss:
            p=candidate;accepted+=1;damping=max(damping/2,1e-8)
            if loss-newloss<1e-9*max(loss,1):break
        else:
            damping=min(damping*5,1e10)
            if damping==1e10:break
    final=np.sum((extended_value_jac(p,u,v,False)-y)**2*weight)
    return p,float(final),dict(iterations=it+1,accepted_steps=accepted)


def fit_extended(target_db,ti,theta,phi,w):
    u,v=log_coordinates(theta,phi,ti);y=target_db.ravel();weight=w.ravel()/w.sum()
    # Fast deterministic initial search, followed by full-grid refinement.
    sel=np.arange(y.size).reshape(w.shape)[::3,::3].ravel()
    coarseweight=weight[sel]/weight[sel].sum()
    starts=[]
    for exponent in (.5,1.5):
        init=np.array([np.max(y),np.log(20),np.log(20),np.log(20),np.log(exponent),np.log(exponent),0.0])
        candidate,_,_=lm_fit(init,u[sel],v[sel],y[sel],coarseweight)
        full_loss=np.sum((extended_value_jac(candidate,u,v,False)-y)**2*weight)
        starts.append((full_loss,candidate))
    p,loss,info=lm_fit(min(starts,key=lambda a:a[0])[1],u,v,y,weight,iterations=25)
    return p,dict(fit_rmse_dB=float(np.sqrt(loss)),**info,
                   bound_hits=int(np.sum(np.isclose(p,LOWER,atol=1e-7)|np.isclose(p,UPPER,atol=1e-7))))


class _Envelope:
    """Seven-parameter water-surface envelope used inside DS+PCA."""

    def __init__(self, theta, phi, weights):
        self.theta, self.phi, self.w = theta, phi, weights

    def fit(self, conditions, maps):
        self.x = kernel_features(conditions)
        parameters = {}
        for key, group in conditions.groupby(KEY, sort=True):
            parameters[key], _ = fit_extended(
                db(maps[group.index]).mean(axis=0), key[2], self.theta, self.phi, self.w)
        targets = np.stack([parameters[tuple(row)] for row in conditions[KEY].to_numpy()])
        self.mean = targets.mean(axis=0)
        self.coef = np.linalg.solve(kernel(self.x, self.x) + 0.001*np.eye(len(conditions)),
                                    targets - self.mean)
        return self

    def predict(self, conditions):
        parameters = self.mean + kernel(kernel_features(conditions), self.x) @ self.coef
        result = []
        for row, parameters_row in zip(conditions.itertuples(), parameters):
            u, v = log_coordinates(self.theta, self.phi, row.theta_i_deg)
            intensity_db = extended_value_jac(np.clip(parameters_row, LOWER, UPPER), u, v, False)
            result.append(np.clip(intensity_db.reshape(self.w.shape), -80, 80))
        return np.stack(result)


class DSPCAModel:
    """Extended DS envelope plus mean residual and three weighted PCA components.

    Both parameter and score regression use ridge coefficient 0.001.
    Predict returns linear scattering intensity on the saved hemisphere grid.
    """
    K = 3
    RIDGE = 0.001

    def __init__(self, theta, phi, weights):
        self.theta, self.phi, self.w = theta, phi, weights

    def fit(self, conditions, maps):
        conditions = conditions.reset_index(drop=True)
        if len(conditions) < self.K + 1:
            raise ValueError("At least four training samples are required for K=3.")
        self.x = kernel_features(conditions)
        self.prior = _Envelope(self.theta, self.phi, self.w).fit(conditions, maps)
        residual = (db(maps) - self.prior.predict(conditions)).reshape(len(conditions), -1)
        self.mean = residual.mean(axis=0)
        self.rootweight = np.sqrt(self.w.ravel() / self.w.sum())
        centred = (residual - self.mean)*self.rootweight
        values, vectors = np.linalg.eigh(centred @ centred.T)
        order = np.argsort(values)[::-1]
        ids = order[:self.K]
        if values[ids[-1]] <= max(values[ids[0]]*1e-12, 1e-12):
            raise ValueError("The training residuals do not support three independent PCA components.")
        self.basis = (vectors[:, ids].T @ centred) / np.sqrt(values[ids])[:, None]
        scores = centred @ self.basis.T
        self.coef = np.linalg.solve(kernel(self.x, self.x) + self.RIDGE*np.eye(len(conditions)), scores)
        self.explained = np.maximum(values[ids], 0) / max(np.maximum(values, 0).sum(), 1e-30)
        return self

    def predict(self, conditions):
        background = self.prior.predict(conditions)
        scores = kernel(kernel_features(conditions), self.x) @ self.coef
        correction = self.mean[None, :] + (scores @ self.basis) / self.rootweight
        intensity_db = background + correction.reshape(len(conditions), len(self.theta), len(self.phi))
        return 10**(np.clip(intensity_db, -80, 100) / 10)

    def save(self, path):
        np.savez_compressed(path, format_version=np.array(1), model_type=np.array("DS+PCA"),
                            k=np.array(self.K), ridge=np.array(self.RIDGE),
                            theta=self.theta, phi=self.phi, w=self.w, x=self.x, mean=self.mean,
                            rootweight=self.rootweight, basis=self.basis, coef=self.coef,
                            explained=self.explained, prior_x=self.prior.x,
                            prior_mean=self.prior.mean, prior_coef=self.prior.coef)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as arrays:
            if (int(arrays["format_version"]) != 1 or str(arrays["model_type"]) != "DS+PCA"
                    or int(arrays["k"]) != cls.K or float(arrays["ridge"]) != cls.RIDGE):
                raise ValueError("Expected the final DS+PCA format with K=3 and ridge=0.001.")
            model = cls(arrays["theta"], arrays["phi"], arrays["w"])
            for name in ["x", "mean", "rootweight", "basis", "coef", "explained"]:
                setattr(model, name, arrays[name])
            if model.basis.shape != (cls.K, model.w.size) or model.coef.shape != (len(model.x), cls.K):
                raise ValueError("Checkpoint basis or regression dimensions are inconsistent.")
            model.prior = _Envelope(model.theta, model.phi, model.w)
            for name in ["x", "mean", "coef"]:
                setattr(model.prior, name, arrays["prior_" + name])
        return model
