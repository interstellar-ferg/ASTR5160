import inspect
from functools import wraps
"""
 AJF chatgpt assisted in creating the following function -
 this creates a wrapper; the wrapper name is placed before a function in other .py files and will read the inputs and outputs of the other function and return a skeleton docstring - 
  makes creating docstrings for functions much faster and easier
  
  for example:
  
  @log_sphinx_io
  def other_func(input1):
      blah
      blah
      
      return blah1
      
 will print
 Paramters:
 ----------
 input1 :class: 'some class'
 
 Returns
 ----------
 :class: 'some class'
 
use with something like:
# AJF import a chat-gpt co-written code that auto-writes docstrings with variables included
from master_scripts.docstring_wrapper import log_sphinx_io as ds
# AJF note: @docstring is a wrapper that auto-writes docstrings for the function directly below it
# AJF see master_scripts/docstring_wrapper for more details
 
"""

def log_sphinx_io(func):
    """Decorator that prints Sphinx-style parameter and return type info."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get argument names and their values
        frame = inspect.currentframe().f_back
        arg_names = inspect.getfullargspec(func).args
        values = dict(zip(arg_names, args))
        values.update(kwargs)

        print(f"\n    Docstring format for function {func.__name__}:\n")
        print("    Parameters\n    ----------")
        for arg in arg_names:
            val = values.get(arg, None)
            if val is not None:
                val_type = type(val)
                if val_type.__module__ == 'builtins':
                    typename = val_type.__name__
                else:
                    typename = f"{val_type.__module__}.{val_type.__name__}"
            else:
                typename = 'Unknown'
            print(f"    {arg} : :class: {typename}\n        ")

        # Call the function and capture the return value(s)
        result = func(*args, **kwargs)

        # Print return types
        print("    Returns\n    ----------")
        if isinstance(result, tuple):
            for i, val in enumerate(result):
                val_type = type(val)
                if val_type.__module__ == 'builtins':
                    typename = val_type.__name__
                else:
                    typename = f"{val_type.__module__}.{val_type.__name__}"
                print(f"    :class: {typename}\n        ")
        else:
            val_type = type(result)
            if val_type.__module__ == 'builtins':
                typename = val_type.__name__
            else:
                typename = f"{val_type.__module__}.{val_type.__name__}"
            print(f"    :class: {typename}\n        ")
        print('\n')
        print('='*150)
        return result

    return wrapper

