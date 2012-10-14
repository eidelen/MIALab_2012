#> ipython --pylab --deep-reload

import dicom
(reader, img) = dicom.open_image("../training/Case1.dcm")

from AgeDetermination import AgeDetermination
aClass = AgeDetermination()

aClass.detect_joints_of_interest(img)

# in ipython run ->
# run fromShell.py
