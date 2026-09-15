flowtorch.rom
=============

``flowtorch.rom`` composes a state-space reduction with a latent regressor.
The initial model set contains continuous-time DMD, continuous-time DMD with
control, and POD interpolation.

Quick start
-----------

.. code-block:: python

   import torch
   from flowtorch.rom import DMD, Trajectory

   model = DMD(rank=20, stationary=True)
   model.fit(Trajectory(snapshots, sample_times))
   prediction = model.predict(
       snapshots[:, 0], time=torch.linspace(0, 1, 101)
   )
   predicted_states = prediction.mean

DMD and DMDc own an :class:`flowtorch.analysis.SVD` directly. Their thin
encoder and decoder views do not contain trainable parameters. The SVD modes,
mean, and weights remain fixed while the continuous latent operator is
fine-tuned. A fitted SVD may be retained while the latent regressor is
reinitialized on new trajectories:

.. code-block:: python

   model.fit_regressor(new_trajectories)

Predictions use vectorized continuous spectral factors. A matrix of initial
states therefore produces a batch of trajectories. DMDc also accepts batched
forcing arrays. Physical time is normalized internally according to
``tau = (t - t0) / time_scale``. The default
``time_scale="sampling_interval"`` makes the training interval one;
alternatively, provide a positive characteristic time or use ``None`` to
disable scaling. Query times and the public ``operator``, ``frequency``,
``growth_rate``, and DMDc ``B`` properties remain in physical units.

Stationary DMDc initialization uses a projected orthogonal Procrustes solution
rather than an iterative alternating fit. At
initialization, ``stationary=True`` therefore produces a proper-orthogonal
latent discrete transition, with unit-modulus eigenvalues, exactly zero
continuous growth rates, and autonomous preservation of the latent Euclidean
norm. Fine-tuning keeps the zero growth rates and unit-modulus spectrum exact,
but its spectral basis is unconstrained; after fine-tuning, the transition is
similar to rotations and need not remain orthogonal or norm preserving. The
option does not mean statistical stationarity, damping, or asymptotic
stability. This neutral spectral constraint is appropriate only when the
modeled autonomous dynamics are expected to have zero growth.

Fine-tuning accepts any PyTorch optimizer class, including
``torch.optim.LBFGS``. Unconstrained dynamics (``stationary=False``) are the
default, but ``stationary_initialization=True`` initializes them from the
stationary proper-orthogonal solution with exactly zero growth. These growth
rates are trainable during fine-tuning. With ``stationary=True``, they instead
remain fixed at zero. Set ``stationary_initialization=False`` to initialize an
unconstrained model from the ordinary least-squares solution. The analytical
fit is retained as the epoch-zero checkpoint, so fine-tuning cannot replace it
unless the selection loss improves. The default rollout objective is
mean-squared error. The default 150-epoch budget uses AdamW with zero weight
decay and a plateau scheduler for 100 stage epochs, followed by 50 stage epochs
of full-batch LBFGS. LBFGS uses strong-Wolfe line search and otherwise retains
the PyTorch defaults, including its internal iteration count and history size.
Both stages optimize all trainable model parameters. Supplying ``optimizer=``
retains single-stage optimization, while ``OptimizationStage`` objects provide
complete control over custom sequences. AdamW uses an early-stopping patience
of 40 epochs by default, while LBFGS uses 10. These can be changed with
``patience`` and ``lbfgs_patience``, respectively.
For DMD and DMDc, fine-tuning uses forward rollout loss only by default.
Backward rollout windows are added only when ``forward_backward=True`` is
passed explicitly.
When noise learning is enabled, early stopping also waits until the relative
change of the complete learned-noise tensor remains below ``1e-4`` for the
same stage patience. The per-epoch values are returned as ``noise_update``;
``noise_update_tolerance`` changes the threshold.
Fine-tuning retains the chronological train/validation split and early stopping
while evaluating trajectory windows as tensor batches. The loss monitored by
scheduling and early stopping is configurable:

.. code-block:: python

   model.fine_tune(
       validation_fraction=0.2,
       validation_loss_weight=0.5,
   )

The monitored loss is ``(1 - validation_loss_weight) * train_loss +
validation_loss_weight * val_loss``. Its default weight is ``0.5``. A weight
of zero monitors training only; a weight of one monitors validation only. If
no complete validation window is available, all windows are used for training
and the weight is ignored. By default, ``validation_fraction=0`` uses all data
for training and scheduling and early stopping monitor the training objective.
Set a positive fraction to request the chronological validation split. Pass
``scheduler=None`` to disable the default scheduler.

Modal-frequency collapse is discouraged by default with a smooth ordering and
spacing penalty. Spacing and temperature are expressed in the physical
frequency units associated with the trajectory time coordinate. Pass an empty
sequence to disable the default penalty, or pass an explicitly configured
instance to override it:

.. code-block:: python

   from flowtorch.rom import FrequencySeparation

   model.fine_tune(regularizers=[])

   model.fine_tune(
       regularizers=[
           FrequencySeparation(
               weight=0.1,
           )
       ]
   )

For real-valued DMD and DMDc models, the penalty acts on the unique non-negative
frequency of each conjugate mode pair. The analytical initialization orders
these pairs. By default, smooth boundary penalties also keep them in the
sampling Nyquist interval. If ``min_spacing`` is omitted, it is one Rayleigh
frequency bin based on the longest fitted trajectory,
``1 / (n_samples * dt)``. An explicit physical spacing and smoothing
temperature can still be supplied. The training log reports data,
regularization, and combined selection losses separately.

Source-backed decompositions reuse the out-of-core and distributed SVD
implementation and return lazy ``StateVectorResult`` predictions.

Time-delay embeddings
---------------------

Higher-order DMD is enabled compositionally by applying a
``TimeDelayEmbedding`` to the POD coordinates. The embedding stacks samples
from oldest to newest and the decoder reads the newest POD state from every
predicted delay vector:

.. code-block:: python

   from flowtorch.rom import DMD, TimeDelayEmbedding, Trajectory

   model = DMD(
       rank=20,
       embedding=TimeDelayEmbedding(4),
   ).fit(Trajectory(snapshots, sample_times))

   prediction = model.predict(
       snapshots[:, :4],
       time=sample_times[3:],
   ).mean

The initial state is a complete physical-state history with shape
``(state, delays)``. Batched histories use ``(state, delays, batch)``. The
POD rank remains available as ``rank`` and the delay-space dimension as
``embedded_rank``.

DMDc can embed the state and control independently. Controls are associated
with time intervals, so a delayed control at the first requested transition
also needs the preceding ``control_delays - 1`` intervals:

.. code-block:: python

   from flowtorch.rom import ControlledTrajectory, DMDc, TimeDelayEmbedding

   model = DMDc(
       rank=20,
       embedding=TimeDelayEmbedding(4),
       control_embedding=TimeDelayEmbedding(4),
   ).fit(ControlledTrajectory(snapshots, sample_times, controls))

   prediction = model.predict(
       snapshots[:, :4],
       time=sample_times[3:],
       forcing=controls[:, 3:],
       forcing_history=controls[:, :3],
   ).mean

State and control delays may differ; fitting aligns both embeddings at the
latest required history sample. Learned measurement noise remains defined in
the unembedded POD coordinates. Consequently, repeated occurrences of the
same snapshot in overlapping delay vectors share exactly one noise parameter.

Monomial embeddings
-------------------

Extended-DMD-style polynomial dictionaries are available with
``MonomialEmbedding``. All monomials from degree one through ``degree`` are
included, including cross terms. The degree-one block is retained so that the
physical-state decoder can read the original POD coordinates exactly. A
constant degree-zero feature can optionally be prepended:

.. code-block:: python

   from flowtorch.rom import DMD, MonomialEmbedding, Trajectory

   model = DMD(
       rank=10,
       embedding=MonomialEmbedding(degree=3, include_constant=True),
   ).fit(Trajectory(snapshots, sample_times))

For example, a two-coordinate state lifted to degree two is ordered as
``[x1, x2, x1**2, x1*x2, x2**2]``, or with a leading one when
``include_constant=True``. The implementation preserves arbitrary batch and
time axes and remains differentiable with respect to its inputs.

The same lift can represent nonlinear forcing dependence in DMDc without
changing the state embedding:

.. code-block:: python

   model = DMDc(
       rank=10,
       control_embedding=MonomialEmbedding(degree=2),
   ).fit(ControlledTrajectory(snapshots, sample_times, controls))

The number of features grows combinatorially with POD rank and degree, so high
degrees are most practical after a suitably aggressive POD reduction.

.. automodule:: flowtorch.rom.embedding
   :members:
   :undoc-members:
   :show-inheritance:

Core interfaces
---------------

.. automodule:: flowtorch.rom.base
   :members:
   :undoc-members:
   :show-inheritance:

SVD encoder and decoder
-----------------------

.. automodule:: flowtorch.rom.svd_encoder
   :members:
   :undoc-members:
   :show-inheritance:

Continuous DMD
--------------

.. automodule:: flowtorch.rom.dmd
   :members:
   :undoc-members:
   :show-inheritance:

Continuous DMD with control
---------------------------

.. automodule:: flowtorch.rom.dmdc
   :members:
   :undoc-members:
   :show-inheritance:

POD interpolation
-----------------

Parameter coordinates are columnwise and may contain any number of
dimensions. For example, steady airfoil snapshots parameterized by angle of
attack and Mach number can be fitted and queried as follows:

.. code-block:: python

   from flowtorch.rom import PODI, ParametricSnapshots

   coordinates = torch.tensor([
       [alpha_1, alpha_2, alpha_3],
       [mach_1,  mach_2,  mach_3],
   ])
   model = PODI(rank=20).fit(
       ParametricSnapshots(flow_snapshots, coordinates)
   )
   prediction = model.predict(
       parameters=torch.tensor([alpha_query, mach_query])
   ).mean

.. automodule:: flowtorch.rom.podi
   :members:
   :undoc-members:
   :show-inheritance:

Fine-tuning and noise
---------------------

Learnable latent measurement noise uses a default regularization weight of
``0.001``. Override ``NoiseConfig(weight=...)`` when a different balance between
rollout accuracy and noise magnitude is required.

The opt-in ``NoiseConfig(penalty="estimated_amplitude")`` mode estimates a
noise standard deviation for every latent channel from a robust short-lag
variogram. Batched Huber regression extrapolates the locally quadratic
variogram to zero lag, and the loss penalizes deviation of the learned-noise
RMS from that estimated scale. The estimator uses PyTorch operations on the
model device and pools differences within, but never across, trajectories.
``variogram_max_lag``, ``variogram_iterations``, and
``variogram_huber_delta`` configure the estimator. The fitted scales are
available as ``model.estimated_noise_std``.
When ``initialization="gaussian"`` is selected, the temporal Gaussian-smoother
residual is rescaled channelwise so that its pooled RMS exactly matches the
variogram estimate. This supplies a structured initial realization at the
estimated amplitude; ``initialization="zeros"`` retains the conservative
zero-noise initialization.

.. automodule:: flowtorch.rom.training
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. automodule:: flowtorch.rom.utils
   :members:
   :undoc-members:
   :show-inheritance:
