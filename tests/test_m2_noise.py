import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import numpy as np
from modules.m2_noise import inject_noise, NOISE_LEVELS
np.random.seed(42)
#for this to pass mean of 10000 samples must be about 0
X_true = [1.02, 0.0, -0.18, 0.0, -0.10, 0.0]
N = 10000
samples = np.array([inject_noise(X_true, t=0.0, level='MEDIUM') for _ in range(N)])
noise= samples-np.array(X_true)  
mean_pos = np.mean(noise[:, 0:3])
mean_vel = np.mean(noise[:, 3:6])
sigma_pos= NOISE_LEVELS['MEDIUM']['sigma_pos']
sigma_vel= NOISE_LEVELS['MEDIUM']['sigma_vel']
threshold_pos= 3*sigma_pos/np.sqrt(N)
threshold_vel= 3*sigma_vel/np.sqrt(N)
print(f"Position noise mean : {mean_pos:.2e}  (must be < {threshold_pos:.2e})")
print(f"Velocity noise mean : {mean_vel:.2e}  (must be < {threshold_vel:.2e})")

if abs(mean_pos) < threshold_pos and abs(mean_vel) < threshold_vel:
    print("\n M2 VALIDATION PASSED — noise is zero-mean and unbiased")
else:
    print("\n M2 VALIDATION FAILED")
    