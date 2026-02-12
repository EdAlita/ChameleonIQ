config
==============

Overview
--------

This package uses `yacs <https://github.com/rbgirshick/yacs>`_ for hierarchical configuration
management, providing a flexible system for managing default settings and user overrides.
For detailed schema, YAML examples, and usage patterns, see :doc:`../guides/configuration`.

Main Components
---------------

.. autosummary::
   config.defaults.get_cfg_defaults

Module Contents
---------------

src.config
~~~~~~~~~~

.. automodule:: config
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

src.config.defaults
~~~~~~~~~~~~~~~~~~~

.. automodule:: config.defaults
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

See Also
--------

- :doc:`../guides/configuration` - Detailed configuration guide
- :doc:`nema_quant` - Main analysis package
- `YACS Documentation <https://github.com/rbgirshick/yacs>`_

Notes
-----

- Configuration is immutable after calling ``.freeze()``
- Always use ``get_cfg_defaults()`` to obtain a fresh copy
- ROI positions use (y, x) convention to match NumPy array indexing
- Hot spheres follow EARL NEMA IQ phantom specification
