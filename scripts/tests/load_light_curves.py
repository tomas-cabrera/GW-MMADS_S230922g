import scripts.utils.light_curves as mylc

###############################################################################

print(mylc.get_light_curve("C202309242206400m275139", "DECam"))
print(mylc.get_light_curve("C202309242206400m275139", "SkyMapper"))
print(mylc.get_light_curve("A202309242250535m165011", "Wendelstein"))
print(mylc.get_light_curve("C202309242206400m275139", "WISE"))
