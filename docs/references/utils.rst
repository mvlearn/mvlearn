Utility
=======

.. currentmodule:: mvlearn.utils

IO
--

.. autofunction:: check_Xs

.. autofunction:: check_Xs_y

.. autofunction:: check_Xs_y_nan_allowed

DCCA Utils
----------

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