"""
Python script for outlier detection based on Deep SVDD.

Copyright (c) 2018 Lukas Ruff
Copyright (C) 2021 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Copyright (c) 2018, Yue Zhao
BSD 2-Clause License

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import torch
from pyod.models.base import BaseDetector
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from torch import nn
from tqdm import tqdm


def get_activation_by_name(name: str) -> nn.Module:
    """Get activation function by name (string)."""

    activations = {
        "relu": nn.ReLU(),
        "prelu": nn.PReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
    }

    if name not in activations.keys():
        raise ValueError(name, "is not a valid activation function")

    return activations[name]


class PyODDataset(torch.utils.data.Dataset):
    """PyOD Dataset class for PyTorch Dataloader."""

    def __init__(self, inputs, mean=None, std=None):
        super().__init__()
        self.inputs = inputs
        self._mean = mean
        self._std = std

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.inputs[idx, :]

        if isinstance(self._mean, np.ndarray) is True:
            sample = (sample - self._mean) / self._std

        return torch.from_numpy(sample), idx


class AutoEncoder(nn.Module):
    """
    Autoencoder for pretraining.
    """

    def __init__(
        self,
        n_features,
        output_dim=32,
        hidden_neurons=None,
        batch_norm=True,
        hidden_activation=None,
        output_activation=None,
    ):
        super().__init__()

        if hidden_neurons is None:
            hidden_neurons = [64, 32]

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.hidden_neurons = hidden_neurons

        # Build encoder
        modules = []
        in_features = n_features
        for _hidden_neurons in self.hidden_neurons:
            modules.append(
                nn.Linear(in_features, out_features=_hidden_neurons, bias=False)
            )
            if batch_norm is True:
                modules.append(nn.BatchNorm1d(_hidden_neurons))

            modules.append(self.hidden_activation)
            in_features = _hidden_neurons
        self.encoder = nn.Sequential(*modules)

        # bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Linear(self.hidden_neurons[-1], output_dim, bias=False),
            self.output_activation,
        )

        # Build decoder
        modules = []
        in_features = output_dim
        for reversed_neurons in self.hidden_neurons[::-1]:
            modules.append(
                nn.Linear(
                    in_features, out_features=reversed_neurons, bias=False
                )
            )
            if batch_norm is True:
                modules.append(nn.BatchNorm1d(reversed_neurons))

            modules.append(self.hidden_activation)
            in_features = reversed_neurons

        modules.append(
            nn.Linear(self.hidden_neurons[0], n_features, bias=False)
        )
        self.decoder = nn.Sequential(*modules)

    def forward(self, x):
        """
        forward.
        """
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


class DeepSVDD_net(nn.Module):
    """
    Network for DeepSVDD.
    """

    def __init__(
        self,
        n_features,
        output_dim=32,
        hidden_neurons=None,
        batch_norm=True,
        dropout_rate=0.0,
        hidden_activation=None,
        output_activation=None,
    ):
        super().__init__()

        if hidden_neurons is None:
            hidden_neurons = [64, 32]

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.hidden_neurons = hidden_neurons
        self.dropout_rate = dropout_rate

        # Build encoder
        modules = []
        in_features = n_features
        """
        for _hidden_neurons in self.hidden_neurons:
            modules.append(
                nn.Linear(in_features, out_features=_hidden_neurons, bias=False)
            )
            if batch_norm is True:
                modules.append(nn.BatchNorm1d(_hidden_neurons))

            modules.append(self.hidden_activation)
            modules.append(nn.Dropout(self.dropout_rate))
            in_features = _hidden_neurons
        """
        modules.append(nn.Linear(in_features, 128, bias=False))
        modules.append(self.hidden_activation)
        modules.append(nn.Linear(128, 64, bias=False))
        modules.append(self.hidden_activation)
        modules.append(nn.Dropout(self.dropout_rate))
        modules.append(nn.Linear(64, 32, bias=False))
        modules.append(self.hidden_activation)
        modules.append(nn.Linear(32, 16, bias=False))
        modules.append(self.hidden_activation)
        modules.append(nn.Dropout(self.dropout_rate))   
        self.encoder = nn.Sequential(*modules)

        # bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Linear(self.hidden_neurons[-1], output_dim, bias=False),
            self.output_activation,
        )

    def forward(self, inputs):
        """
        Perform forward propagation.
        """
        hidden = self.encoder(inputs)
        embed = self.bottleneck(hidden)
        return embed


class DeepSVDD(BaseDetector):
    """Deep One-Class Classifier with AutoEncoder (AE) is a type of neural
    networks for learning useful data representations unsupervisedly.
    Similar to PCA, DeepSVDD could be used to detect outlying objects
    in the data by calculating the distance from center.
    See :cite:`ruff2018deepsvdd` for details.

    Parameters
    ----------
    objective : str, optional (default='soft-boundary')
        A string specifying the Deep SVDD objective
        either 'one-class' or 'soft-boundary'.

    c: float, optional (default=None)
        Deep SVDD center, the default will be calculated based on network
        initialization first forward pass. To get repeated results set
        random_state if c is set to None.

    nu: float, optional (default=0.1)
        Deep SVDD hyperparameter nu (must be 0 < nu <= 1).

    hidden_neurons : list, optional (default=[64, 32])
        The number of neurons per hidden layers.

    hidden_activation : str, optional (default='relu')
        Activation function to use for hidden layers.
        All hidden layers are forced to use the same type of activation.

    output_dim : int, optional (default=32)
        The number of neurons at output layers of Deep SVDD.

    output_activation : str, optional (default='sigmoid')
        Activation function to use for output layer.

    epochs : int, optional (default=50)
        Number of epochs to train the model.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    dropout_rate : float in (0., 1), optional (default=0.2)
        The dropout to be used across all layers.

    leaning_rate : float in (0., 1), optional (default=0.001)
        Learning rate to be used in updating network weights.

    weight_decay : float in (0., 1), optional (default=0.1)
        The regularization strength of activity_regularizer
        applied on each layer.

    validation_size : float in (0., 1), optional (default=0.1)
        The percentage of data to be used for validation.

    preprocessing : bool, optional (default=False)
        If True, apply standardization on the data.

    pretraining : bool, optional (default=False)
        If True, an autoencoder is pretrained.
        The network weights of Deep SVDD will be initialized
        by using pre-trained weights from the autoencoder.

    pretrain_epochs : int, optional (default=10)
        Number of epochs to pre-train the autoencoder.

    warm_up_epochs : int, optional (default=10)
        Number of training epochs for soft-boundary Deep SVDD
        before radius R gets updated.

    batch_norm : bool, optional (default=True)
        If True, apply standardization on the data.

    criterion : Torch Module, optional (default=torch.nn.MSEloss)
        A criterion that measures erros between
        network output and the Deep SVDD center.

    verbose : int, optional (default=0)
        Verbosity mode.

        - 0 = silent
        - 1 = progress bar
        - 2 = one line per epoch.

        For verbose >= 1, model summary may be printed.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. When fitting this is used
        to define the threshold on the decision function.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(
        self,
        objective="soft-boundary",
        c=None,
        nu=0.1,
        hidden_neurons=None,
        hidden_activation="relu",
        output_dim=32,
        output_activation="sigmoid",
        epochs=50,
        batch_size=32,
        dropout_rate=0.2,
        learning_rate=1e-3,
        weight_decay=1e-5,
        validation_size=0.1,
        preprocessing=False,
        pretraining=False,
        pretrain_epochs=10,
        warm_up_epochs=10,
        batch_norm=True,
        criterion=torch.nn.MSELoss(),
        verbose=0,
        contamination=0.1,
        device=None,
    ):
        super().__init__(contamination=contamination)

        assert objective in (
            "one-class",
            "soft-boundary",
        ), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        assert (nu > 0) & (
            nu <= 1
        ), "For hyperparameter nu, it must hold: 0 < nu <= 1."
        self.nu = nu

        if hidden_neurons is None:
            hidden_neurons = [64, 32]

        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.R = torch.tensor(0.0, device=self.device)
        self.c = torch.tensor(c, device=self.device) if c is not None else None

        self.hidden_neurons = hidden_neurons
        self.hidden_activation = get_activation_by_name(hidden_activation)
        self.output_dim = output_dim
        self.output_activation = get_activation_by_name(output_activation)
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.pretraining = pretraining
        self.pretrain_epochs = pretrain_epochs
        self.warm_up_n_epochs = warm_up_epochs
        self.batch_norm = batch_norm
        self.criterion = criterion
        self.verbose = verbose

        self._ae_net = None
        self._svdd_net = None
        self.decision_scores_ = None
        self._mean = None
        self._std = None

    def _init_center_c(self, train_loader, eps=0.1):
        """
        Initialize hypersphere center c as the mean
        from an initial forward pass on the data.
        """

        c = torch.zeros(self.output_dim, device=self.device)

        n_samples = 0
        self._svdd_net.eval()
        with torch.no_grad():
            for data, _ in train_loader:
                inputs = data.to(self.device).float()
                outputs = self._svdd_net(inputs)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps.
        # Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

    def _train_AutoEncoder(self, train_loader):
        """
        Internal function to train AutoEncoder.

        Parameters
        ----------
        train_loader : torch dataloader
            Data loader of training data.
        """

        optimizer = torch.optim.Adam(
            self._ae_net.parameters(), lr=self.learning_rate, weight_decay=0.0
        )

        self._ae_net.train()
        for _ in range(self.pretrain_epochs):
            training_loss = 0.0
            for data, _ in train_loader:
                inputs = data.to(self.device).float()
                optimizer.zero_grad()
                outputs = self._ae_net(inputs)
                loss = self.criterion(inputs, outputs)
                loss.backward()
                training_loss += loss.item()
                optimizer.step()

    def _init_network_weights_from_pretraining(self):
        """
        Initialize the Deep SVDD network weights from the encoder
        weights of the pretraining autoencoder.
        """

        net_dict = self._svdd_net.state_dict()
        ae_net_dict = self._ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}

        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)

        # Load the new state_dict
        self._svdd_net.load_state_dict(net_dict)

    def _get_loss(self, dist):
        """
        Internal function to compute loss.
        """
        if self.objective == "soft-boundary":
            scores = dist - self.R ** 2
            loss = self.R ** 2 + (1 / self.nu) * torch.mean(
                torch.max(torch.zeros_like(scores), scores)
            )
        else:  # one-class deep SVDD
            loss = torch.mean(dist)

        return loss

    def _update_radius(self, dist, epoch):
        """
        Internal function to update radius R.
        Optimally solve for radius R via the (1-nu)-quantile of distances.
        """
        if (self.objective == "soft-boundary") and (
            epoch >= self.warm_up_n_epochs
        ):
            newR = np.quantile(
                np.sqrt(dist.cpu().detach().numpy()), 1 - self.nu
            )
            self.R.data = torch.tensor(newR, device=self.device)

    def _train_DeepSVDD(self, train_loader, val_loader):
        """
        Internal function to train DeepSVDD.

        Parameters
        ----------
        train_loader : torch dataloader
            Data loader of training data.

        val_loader : torch dataloader
            Data loader of validation data.
        """

        optimizer = torch.optim.Adam(
            self._svdd_net.parameters(), lr=self.learning_rate
        )

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            self.c = self._init_center_c(train_loader, eps=0.01)

        tqdm_disable = True
        if self.verbose == 1:
            tqdm_disable = False

        for epoch in tqdm(range(self.epochs), disable=tqdm_disable):
            training_loss = []
            self._svdd_net.train()
            for data, _ in train_loader:
                inputs = data.to(self.device).float()
                optimizer.zero_grad()

                # output of Deep SVDD net (embedded represention)
                embed = self._svdd_net(inputs)

                # distance from center
                dist = torch.sum((embed - self.c) ** 2, dim=1)

                # compute objective function (loss)
                loss = self._get_loss(dist)

                # add weight decay (L2)
                decay = torch.tensor(0.0, requires_grad=True)
                for w in self._svdd_net.parameters():
                    decay = decay + torch.norm(w) ** 2
                loss = loss + self.weight_decay * decay

                # update weights
                loss.backward()
                optimizer.step()
                training_loss.append(loss.item())

                # update radius R
                self._update_radius(dist, epoch)

            if len(val_loader) > 0:
                self._svdd_net.eval()
                val_loss = []
                with torch.no_grad():
                    for data, _ in val_loader:
                        inputs = data.to(self.device).float()
                        embed = self._svdd_net(inputs)  # [N, D]
                        dist = torch.sum((embed - self.c) ** 2, dim=1)  # [N, 1]
                        loss = self._get_loss(dist)
                        val_loss.append(loss.item())

            if len(val_loader) > 0 and self.verbose == 2:
                print(
                    "Epoch {}/{}: train_loss={:.6f}, val_loss={:.6f}".format(
                        epoch + 1,
                        self.epochs,
                        np.mean(training_loss),
                        np.mean(val_loss),
                    )
                )
            elif self.verbose == 2:
                print(
                    "Epoch {}/{}: loss={:.6f}".format(
                        epoch + 1, self.epochs, np.mean(training_loss)
                    )
                )

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        # Verify and construct the hidden units
        n_features = X.shape[1]

        # make dataset and dataloader
        # conduct standardization if needed
        if self.preprocessing:
            self._mean, self._std = np.mean(X, axis=0), np.std(X, axis=0)
            dataset = PyODDataset(inputs=X, mean=self._mean, std=self._std)
        else:
            dataset = PyODDataset(inputs=X)

        train_size = int(len(dataset) * (1.0 - self.validation_size))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        # Initialize Deep SVDD
        self._svdd_net = DeepSVDD_net(
            n_features=n_features,
            output_dim=self.output_dim,
            hidden_neurons=self.hidden_neurons,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            hidden_activation=self.hidden_activation,
            output_activation=self.output_activation,
        )
        self._svdd_net = self._svdd_net.to(self.device)

        # pre-training using autoencoder
        if self.pretraining is True:

            # initialize autoencoder
            self._ae_net = AutoEncoder(
                n_features=n_features,
                output_dim=self.output_dim,
                hidden_neurons=self.hidden_neurons,
                batch_norm=self.batch_norm,
                hidden_activation=self.hidden_activation,
                output_activation=self.output_activation,
            )
            self._ae_net = self._ae_net.to(self.device)

            # perform training
            self._train_AutoEncoder(train_loader)

            # copy weights from AE to Deep SVDD
            self._init_network_weights_from_pretraining()

        # perform training of Deep SVDD
        self._train_DeepSVDD(train_loader, val_loader)

        self.decision_scores_ = self.decision_function(X)
        self._process_decision_scores()

        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        outlier_scores : numpy array of shape (n_samples,)
            The outlier score of the input samples.
        """
        check_is_fitted(self, ["_svdd_net"])
        X = check_array(X)

        if self.preprocessing:
            # self._mean, self._std = np.mean(X, axis=0), np.std(X, axis=0)
            valid_set = PyODDataset(inputs=X, mean=self._mean, std=self._std)
        else:
            valid_set = PyODDataset(inputs=X)

        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        # enable the evaluation mode
        self._svdd_net.eval()

        outlier_scores = []
        with torch.no_grad():
            for data, _ in valid_loader:
                inputs = data.to(self.device).float()
                embed = self._svdd_net(inputs)
                dist = torch.sum((embed - self.c) ** 2, dim=1)

                score = dist.to("cpu").detach().numpy().copy()
                outlier_scores.append(score)

        outlier_scores = np.concatenate(outlier_scores)
        return outlier_scores
