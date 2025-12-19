import numpy as np
import matplotlib.pyplot as plt
import miepython

# ==========================================================
# Calculating Gamma Droplet Size Distribution
# Based on ref: https://www.rand.org/content/dam/rand/pubs/reports/2006/R456.pdf
# ==========================================================
def gamma_dsd(r, N_tot, r_mode, mu):
    """
    Returns a Gamma droplet size distribution n(r) in #/(m^3 * m).
    Based on the 'modified gamma' formulation common in cloud physics.

    Parameters:
    r : array
        Droplet radii (meters).
    N_tot : float
        Total droplet number concentration (#/m^3).
    r_mode : float
        Mode radius, the radius where n(r) peaks (meters).
    mu : float
        Shape parameter (unitless). Higher mu = narrower distribution.
        Typical values for fog/clouds: mu = 2 to 6.

    Returns:
    n_r : array
        Number concentration density per radius interval (#/m^4).
    """
    # Calculate the scale parameter lambda from r_mode and mu
    # For the gamma function, the mode is at r_mode = (mu - 1) / lambda
    lam = (mu - 1) / r_mode

    # Normalization constant to ensure ∫ n(r) dr = N_tot
    # Using the analytical integral of the gamma function
    from scipy.special import gamma
    C = N_tot * (lam**mu) / gamma(mu)

    # Gamma distribution: n(r) = C * r^(mu-1) * exp(-lam * r)
    n_r = C * (r**(mu - 1)) * np.exp(-lam * r)
    return n_r

# ==========================================================
# Finding correct r_mode to match both N_tot and LWC
# ==========================================================

def find_correct_r_mode(N_tot, LWC_target, mu, radii, water_density=1e6):
    """
    Search for r_mode that makes the DSD match both N_tot and LWC_target.
    Returns: best_r_mode, n_r (corrected distribution)
    """
    best_r_mode = None
    best_error = 1e9
    best_n_r = None
    
    # Search over plausible r_mode values
    r_mode_candidates = np.linspace(2e-6, 15e-6, 27)  # 2 to 15 µm in 0.5 µm steps
    
    for r_mode_candidate in r_mode_candidates:
        # Create DSD with this r_mode
        n_r_candidate = gamma_dsd(radii, N_tot, r_mode_candidate, mu)
        
        # Calculate actual LWC from this DSD
        volume_integral = np.trapezoid(n_r_candidate * (4/3) * np.pi * radii**3, radii)
        LWC_calculated = water_density * volume_integral
        
        # Calculate error (how far from target LWC)
        LWC_error = abs(LWC_calculated - LWC_target) / LWC_target
        
        # Also check N_tot match
        N_calculated = np.trapezoid(n_r_candidate, radii)
        N_error = abs(N_calculated - N_tot) / N_tot
        
        total_error = LWC_error + N_error
        
        if total_error < best_error:
            best_error = total_error
            best_r_mode = r_mode_candidate
            best_n_r = n_r_candidate
    
    # Final verification
    final_LWC = water_density * np.trapezoid(best_n_r * (4/3) * np.pi * radii**3, radii)
    final_N = np.trapezoid(best_n_r, radii)
    
    print(f"\n=== DSD OPTIMIZATION ===")
    print(f"Found r_mode = {best_r_mode*1e6:.1f} µm")
    print(f"Achieved N_tot = {final_N:.2e} m⁻³ (target: {N_tot:.2e})")
    print(f"Achieved LWC = {final_LWC:.3f} g/m³ (target: {LWC_target:.3f})")
    
    return best_r_mode, best_n_r

# ==========================================================
# 1. Refractive indices (8–14 µm)
# ref: https://refractiveindex.info/?shelf=main&book=H2O&page=Warren-2008
# ==========================================================

# Water at 25 °C
water_25C = {
    8e-6: (1.2681, 0.034267),
    9e-6: (1.22377, 0.039920),
    10e-6: (1.1932, 0.050791),
    11e-6: (1.1280, 0.097402),
    12e-6: (1.0875, 0.19956),
    13e-6: (1.1220, 0.30591),
    14e-6: (1.1850, 0.36981),
}

# Water at 0 °C
water_0C = {
    8e-6: (1.2952, 0.036344),
    9e-6: (1.2619, 0.039432),
    10e-6: (1.2090, 0.053098),
    11e-6: (1.1345, 0.10816),
    12e-6: (1.1033, 0.23757),
    13e-6: (1.1516, 0.35367),
    14e-6: (1.2361, 0.42545),
}

# Ice at -7 °C
ice_C = {
    8e-6: (1.3067, 0.037900),
    9e-6: (1.2700, 0.036824),
    10e-6: (1.1926, 0.050080),
    11e-6: (1.0886, 0.24800),
    12e-6: (1.2762, 0.41333),
    13e-6: (1.4697, 0.38812),
    14e-6: (1.5657, 0.28068),
}

materials = {
    "Water 25°C": water_25C,
    "Water 0°C":  water_0C,
    "Ice -7°C":    ice_C
}

wavelengths = np.array([8e-6, 9e-6, 10e-6, 11e-6, 12e-6, 13e-6, 14e-6]) #LWIR range in meters

# ==========================================================
# 2. Fog parameters
'''
 ref LWC: https://link.springer.com/article/10.1007/s00024-007-0211-x
          and fig3a https://www-pm.larc.nasa.gov/icing/pub/conf/Gultepe.etal.eabs.AMS.09.pdf
 ref r_mode/mu: https://www.rand.org/content/dam/rand/pubs/reports/2006/R456.pdf
 ref visibility categories: https://skybrary.aero/articles/precision-approach
Light:         Visibility 0.5-1 km,    LWC = 0.05-0.12
Moderate:      Visibility 0.3-0.5 km,  LWC = 0.1-0.2
Dense:         Visibility 0.2-0.3 km,  LWC = 0.15-0.3
Very dense:    Visibility 0.05-0.2 km, LWC = 0.25-0.5
Extreme dense: Visibility < 0.05 km,   LWC = > 0.4
'''
# ==========================================================
# Target physical properties from visibility and LWC
Vis_target = 0.2  # km
LWC_target = 0.25 # g/m^3, chosen from plausible range
water_density = 1e6  # g/m^3, standard density of water

# 1. Calculate the consistent N_d (droplets density) 
# using Equation 1 from https://www-pm.larc.nasa.gov/icing/pub/conf/Gultepe.etal.eabs.AMS.09.pdf
N_d_per_cm3 = ((1.002 / Vis_target) ** (1 / 0.6473)) / LWC_target 
N_tot = N_d_per_cm3 * 1e6  # Convert to droplets per m³

print(f"For Vis={Vis_target} km, LWC={LWC_target} g/m³:")
print(f"Consistent N_d = {N_d_per_cm3:.1f} cm⁻³ = {N_tot:.2e} m⁻³")

# 2. Define your DSD (Droplet Size Distribution) parameters (these are choices based on fog type)
# ref: https://www.sciencedirect.com/science/article/pii/S0169809524003521#:~:text=Smaller%20droplets%20(1%2D10%20%CE%BCm,volume%20diameter%20of%2024.8%20%C2%B5m.
radii = np.linspace(0.5e-6, 25e-6, 200) # droplet radii from 0.5 µm to 25 µm in meters
path_length = Vis_target * 1000       # meters

mu = 3  # (choose 2-6) Controls width or spread of the distribution of the fog around the r_mode
# Mode radius = (choose 2-8 µm) Most common droplet size in the fog
# n_r is ow many droplets of each size exist per cubic meter of fog
r_mode, n_r = find_correct_r_mode(N_tot, LWC_target, mu, radii, water_density)

# ==========================================================
# 3. Mie calculations
# ==========================================================

def compute_Q(material_indices):
    results = {}
    for wl in wavelengths:
        n, k = material_indices[wl]
        m = complex(n, k)

        Qext_list, Qsca_list, Qabs_list = [], [], []

        for r in radii:
            x = 2 * np.pi * r / wl
            qext, qsca, qback, g = miepython.efficiencies_mx(m, x)

            Qext_list.append(qext)
            Qsca_list.append(qsca)
            Qabs_list.append(qext - qsca)

        results[wl] = {
            "Qext": np.array(Qext_list),
            "Qsca": np.array(Qsca_list),
            "Qabs": np.array(Qabs_list),
        }

    return results

results = {name: compute_Q(table) for name, table in materials.items()}

# ==========================================================
# 4. Fog transmission
# ref: Beer Lambert Law: T = exp(-αL), α = N_d * σ_ext, σ_ext = (pi*r**2*Qext)
# ==========================================================

def sigma_ext(Qext, r):
    return np.pi * r**2 * Qext

transmissions = {}

for name, table in materials.items():
    Tvals = []
    
    for wl in wavelengths:
        # Get Qext values for ALL radii at this wavelength
        Qext_values = results[name][wl]["Qext"]  
        
        # Calculate extinction cross-section for each radius
        sigma_values = sigma_ext(Qext_values, radii)  
        
        # Integrate over the DSD: α = ∫ n(r) * σ(r) dr
        integrand = n_r * sigma_values  # n(r) * σ(r)
        alpha = np.trapezoid(integrand, radii)  # Total extinction coefficient (m⁻¹)
        
        # Beer-Lambert law
        T = np.exp(-alpha * path_length)
        Tvals.append(T)

    transmissions[name] = np.array(Tvals)

# ==========================================================
# 5. Plotting
# ==========================================================
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Droplet Size Distribution
axs[0, 0].plot(radii * 1e6, n_r / 1e6)  # Convert to #/(cm³·µm) for readability
axs[0, 0].set_title(f"Gamma DSD (r_mode={r_mode*1e6:.1f}µm, μ={mu})")
axs[0, 0].set_xlabel("Droplet Radius (µm)")
axs[0, 0].set_ylabel("n(r) (#/cm³·µm)")
axs[0, 0].grid(True)
axs[0, 0].axvline(r_mode*1e6, color='r', linestyle='--', label=f'Mode: {r_mode*1e6:.1f}µm')
axs[0, 0].legend()

# Plot 2: Qext vs Radius (at a sample wavelength)
sample_wl = 11e-6  # specific Infrared light
sample_idx = np.argmin(np.abs(wavelengths - sample_wl))
for name in materials.keys():
    axs[0, 1].plot(radii * 1e6, results[name][wavelengths[sample_idx]]["Qext"], label=name)
axs[0, 1].set_title(f"Qext at λ={sample_wl*1e6:.0f}nm")
axs[0, 1].set_xlabel("Droplet Radius (µm)")
axs[0, 1].set_ylabel("Qext")
axs[0, 1].grid(True)
axs[0, 1].legend()

# Plot 3: Fog Transmission Spectrum
for name in materials.keys():
    axs[1, 0].plot(wavelengths * 1e6, transmissions[name], label=name)
axs[1, 0].set_title(f"Fog Transmission\nVision={Vis_target*1000:.0f}m, LWC={LWC_target}g/m³")
axs[1, 0].set_xlabel("Wavelength (µm)")
axs[1, 0].set_ylabel("Transmission")
axs[1, 0].grid(True)
axs[1, 0].legend()

# Plot 4: Extinction coefficient spectrum
material_for_extinction = "Water 25°C"
if material_for_extinction in transmissions:
    axs[1, 1].plot(wavelengths * 1e6, -np.log(transmissions[material_for_extinction]) / path_length, 'k-')
    axs[1, 1].set_title("Extinction Coefficient α(λ)")
    axs[1, 1].set_xlabel("Wavelength (µm)")
    axs[1, 1].set_ylabel("α (m⁻¹)")
    axs[1, 1].grid(True)

plt.tight_layout()
plt.show()