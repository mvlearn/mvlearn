mvlearn.embed
===============

.. currentmodule:: mvlearn.embed

Generalized Canonical Correlation Analysis
------------------------------------------

.. autoclass:: GCCA
    :exclude-members: get_params, set_params

Kernel Canonical Correlation Analysis
-------------------------------------

.. autoclass:: KCCA
    :exclude-members: get_params, set_params

Deep Canonical Correlation Analysis
-----------------------------------

.. autoclass:: DCCA
    :exclude-members: get_params, set_params

Omnibus Embedding
-----------------

.. autoclass:: Omnibus
    :exclude-members: transform, get_params, set_params

Partial Least Squares Regression
--------------------------------

.. autofunction:: partial_least_squares_embedding

Multiview Multidimensional Scaling
----------------------------------

.. autoclass:: MVMDS
    :exclude-members: transform, get_params, set_params

Split Autoencoder
-----------------

.. autoclass:: SplitAE
    :exclude-members: get_params, set_params

DCCA Utilities
--------------

.. autoclass:: linear_cca

.. autoclass:: cca_loss

.. autoclass:: MlpNet
	:exclude-members: add_module, apply, buffers, children, named_children,
		named_modules, named_parameters, register_backward_hook,
		register_buffer, register_forward_hook, register_forward_pre_hook,
		register_parameter, requires_grad, train, type, eval, extra_repr,
		double, float, half, named_buffers, zero_grad, to, state_dict,
		requires_grad, modules, cpu, cuda, load_state_dict

.. autoclass:: DeepCCA
	:exclude-members: add_module, apply, buffers, children, named_children,
		named_modules, named_parameters, register_backward_hook,
		register_buffer, register_forward_hook, register_forward_pre_hook,
		register_parameter, requires_grad, train, type, eval, extra_repr,
		double, float, half, named_buffers, zero_grad, to, state_dict,
		requires_grad, modules, cpu, cuda, load_state_dict
