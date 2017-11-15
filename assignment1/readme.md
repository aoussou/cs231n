NOTE: line 121 of features.py was change to account for for compatibility issues with Python 3.6

Namely

    orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(cx, cy))[cx/2::cx,cy/2::cy].T
    
was replaced with 

    orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=(cx, cy))[int(cx/2)::cx, int(cy/2)::cy].T
