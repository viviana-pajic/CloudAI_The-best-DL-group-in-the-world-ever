"""GRU training and CPU inference. Scalers and epoch selection use training data."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import time
import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from research_models import SEED, metric
from weather_search import BASE_WEATHER
from common.hardware import gpu_lock, deadline_passed
CONTEXT = ['weekday_sin','weekday_cos','year_sin','year_cos','weekend','holiday','before_holiday','after_holiday','trend']
torch.set_num_threads(2)


class SequenceNetwork(nn.Module):
    def __init__(self,architecture,hidden,head,context_features,dropout):
        super().__init__()
        self.architecture=architecture
        if architecture=="gru":
            self.sequence=nn.GRU(1,hidden,batch_first=True)
        else:
            self.convolutions=nn.ModuleList([nn.Conv1d(1 if number==0 else hidden,hidden,3,dilation=dilation,padding=2*dilation) for number,dilation in enumerate([1,2,4,8])])
        self.dropout=nn.Dropout(dropout)
        self.head=nn.Sequential(nn.Linear(hidden+context_features,head),nn.ReLU(),nn.Dropout(dropout),nn.Linear(head,1))

    def forward(self,sequence,context):
        if self.architecture=="gru":
            values,_=self.sequence(sequence)
            summary=values[:,-1]
        else:
            values=sequence.transpose(1,2)
            for convolution in self.convolutions:
                values=torch.relu(convolution(values))[...,:sequence.shape[1]]
                values=self.dropout(values)
            summary=values[:,:,-1]
        return self.head(torch.cat([summary,context],dim=1)).squeeze(-1)

class SequenceRegressor(RegressorMixin,BaseEstimator):
    def __init__(self,architecture="gru",hidden=16,head=8,dropout=.15,learning_rate=.003,weight_decay=.001,max_epochs=180,patience=20,target="logratio7",weather=False,loss="huber",batch_size=64,seed=SEED):
        self.architecture=architecture
        self.hidden=hidden
        self.head=head
        self.dropout=dropout
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.max_epochs=max_epochs
        self.patience=patience
        self.target=target
        self.weather=weather
        self.loss=loss
        self.batch_size=batch_size
        self.seed=seed

    def _base(self,X):
        return X.lag_7.to_numpy() if self.target=="logratio7" else X.mean_28.to_numpy()

    def _sequence(self,X):
        values=X[[f"lag_{n}" for n in range(28,0,-1)]].to_numpy()
        prior=X.mean_28.to_numpy()
        return (np.log1p(values)-np.log1p(prior[:,None])).astype(np.float32)[...,None]

    def _new_model(self,context_features,device):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True,warn_only=True)
        return SequenceNetwork(self.architecture,self.hidden,self.head,context_features,self.dropout).to(device)

    def _arrays(self,X,y,context_scaler,target_scaler,device):
        sequence=torch.tensor(self._sequence(X),device=device)
        context=torch.tensor(context_scaler.transform(X[self.context_columns_]).astype(np.float32),device=device)
        target=np.log1p(np.asarray(y))-np.log1p(self._base(X))
        target=torch.tensor(target_scaler.transform(target.reshape(-1,1)).ravel().astype(np.float32),device=device)
        return sequence,context,target

    def _epoch(self,network,optimizer,arrays):
        network.train()
        sequence,context,target=arrays
        order=torch.randperm(len(target),device=target.device)
        total=0.
        for start in range(0,len(target),self.batch_size):
            index=order[start:start+self.batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction=network(sequence[index],context[index])
            loss=(torch.nn.functional.l1_loss(prediction,target[index]) if self.loss=="l1" else torch.nn.functional.smooth_l1_loss(prediction,target[index],beta=.5))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),5.)
            optimizer.step()
            total+=float(loss.detach())*len(index)
        return total/len(target)

    def _restore(self,values,X,scaler):
        target=scaler.inverse_transform(np.asarray(values).reshape(-1,1)).ravel()
        return np.maximum(0,np.expm1(np.clip(target+np.log1p(self._base(X)),0,16)))

    def fit(self,X,y):
        if not torch.cuda.is_available():
            raise RuntimeError("The authorized sequence experiment requires verified CUDA.")
        if deadline_passed():
            raise TimeoutError("Research time budget exhausted")
        self.context_columns_=CONTEXT+(BASE_WEATHER if self.weather else [])
        inner_split=len(X)-28
        assert inner_split>=60
        inner_X=X.iloc[:inner_split]
        inner_y=np.asarray(y)[:inner_split]
        inner_check=X.iloc[inner_split:]
        inner_actual=np.asarray(y)[inner_split:]
        self.inner_split_={"training_rows":inner_split,"validation_rows":28,"training_end":str(inner_X.index[-1].date()),"validation_start":str(inner_check.index[0].date()),"validation_end":str(inner_check.index[-1].date())}
        began=time.perf_counter()
        with gpu_lock("citibike-sequence:"+self.architecture):
            device="cuda"
            context_scaler=StandardScaler().fit(inner_X[self.context_columns_])
            targets=np.log1p(inner_y)-np.log1p(self._base(inner_X))
            target_scaler=StandardScaler().fit(targets.reshape(-1,1))
            arrays=self._arrays(inner_X,inner_y,context_scaler,target_scaler,device)
            check_sequence=torch.tensor(self._sequence(inner_check),device=device)
            check_context=torch.tensor(context_scaler.transform(inner_check[self.context_columns_]).astype(np.float32),device=device)
            network=self._new_model(len(self.context_columns_),device)
            optimizer=torch.optim.AdamW(network.parameters(),lr=self.learning_rate,weight_decay=self.weight_decay)
            best=float("inf")
            best_epoch=1
            history=[]
            for epoch in range(1,self.max_epochs+1):
                if deadline_passed():
                    raise TimeoutError("Time budget exhausted during neural fitting")
                training_loss=self._epoch(network,optimizer,arrays)
                network.eval()
                with torch.inference_mode():
                    values=network(check_sequence,check_context).cpu().numpy()
                predicted=self._restore(values,inner_check,target_scaler)
                score=metric(inner_actual,predicted)["mae"]
                history.append({"epoch":epoch,"training_loss":training_loss,"inner_mae":score})
                if score<best-1.:
                    best=score
                    best_epoch=epoch
                if epoch-best_epoch>=self.patience:
                    break
            self.epochs_=best_epoch
            self.inner_history_=history
            self.inner_best_mae_=best
            # Reinitialize and fit all outer training rows. Outer CV targets are
            # not used for early stopping, scaling, epoch choice or updates.
            self.context_scaler_=StandardScaler().fit(X[self.context_columns_])
            all_targets=np.log1p(np.asarray(y))-np.log1p(self._base(X))
            self.target_scaler_=StandardScaler().fit(all_targets.reshape(-1,1))
            all_arrays=self._arrays(X,y,self.context_scaler_,self.target_scaler_,device)
            network=self._new_model(len(self.context_columns_),device)
            optimizer=torch.optim.AdamW(network.parameters(),lr=self.learning_rate,weight_decay=self.weight_decay)
            for epoch in range(self.epochs_):
                if deadline_passed():
                    raise TimeoutError("Time budget exhausted during outer refit")
                self._epoch(network,optimizer,all_arrays)
            self.network_=network.cpu().eval()
            self.training_device_=str(torch.cuda.get_device_name(0))
            self.parameter_count_=sum(parameter.numel() for parameter in network.parameters())
            del arrays,all_arrays,check_sequence,check_context,optimizer
            torch.cuda.empty_cache()
        self.fit_seconds_=time.perf_counter()-began
        return self

    def predict(self,X):
        sequence=torch.tensor(self._sequence(X),device="cpu")
        context=torch.tensor(self.context_scaler_.transform(X[self.context_columns_]).astype(np.float32),device="cpu")
        with torch.inference_mode():
            values=self.network_(sequence,context).numpy()
        return self._restore(values,X,self.target_scaler_)

def sequence_estimator(configuration):
    return SequenceRegressor(**configuration)
