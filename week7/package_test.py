# AJF created 3/15/25
# last modified by AJF on 3/15/25

# AJF using scheme of 'import location.module import function'
from week5.sphere_area import area
print(f'using scheme of import location.module import function')
area()

# AJF using scheme of 'import location.module import function as localname'
from week5.sphere_area import area as ar
print(f'\n\n\n\nusing scheme of import location.module import function as localname')
ar()

# AJF using scheme import location.module, then location.module.function (could also...
# ... do location.module as localname, then use localname.function() as call
import week5.sphere_area
print(f'\n\n\n\nusing scheme import location.module, then module.function')
week5.sphere_area.area()
