{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :member-order: bysource

   {% block methods %}
   {% if methods %}

   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :toctree:
   {% for item in all_methods %}
      {%- if not item.startswith('_') or item in ['__call__'] %}
      {{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}

   {% endif %}
   {% endblock %}



   {% block attributes %}
   {% if attributes %}

   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
      :toctree:
   {% for item in attributes %}
      {{ name }}.{{ item }}
   {%- endfor %}

   {% endif %}
   {% endblock %}
